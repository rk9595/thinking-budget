"""Frozen, inexpensive SFT learnability ablation; training-set success is not a release gate."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from compare_pilot import load_evaluation
from compare_results import paired_difference, reference_gate
from eval_budget import paired_control, summarize
from experiment import digest, environment, read_records, write_json

BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"


def select_ids(records, seed=73, train_count=24, probe_count=12):
    ids = sorted({r["problem_id"] for r in records})
    if len(ids) < train_count + probe_count:
        raise ValueError("not enough distinct verified problems")
    random.Random(seed).shuffle(ids)
    return ids[:train_count], ids[train_count:train_count+probe_count]


def train_gate(rows, base):
    checks = reference_gate(rows, base)["checks"]
    checks["training_accuracy_at_least_75pct_each_budget"] = all(
        r["accuracy"] >= .75 for r in summarize(rows))
    return {"passed": all(checks.values()), "checks": checks,
            "note": "Overfitting/learnability diagnostic on 24 training problems, not held-out success."}


def analyze(root):
    plan = json.loads((root/"plan.json").read_text())
    loaded = {name: load_evaluation(root/(name+".json")) for name in ["base", "token", "sequence"]}
    bm = loaded["base"][1]
    expected = {"budgets": plan["budgets"], "seeds": [plan["seed"]],
                "batch_size": plan["batch_size"], "max_tokens": plan["max_tokens"]}
    if bm["model_revision"] != plan["revision"]:
        raise ValueError("evaluation model differs from frozen plan")
    for key, value in expected.items():
        if bm["config"][key] != value:
            raise ValueError(f"evaluation differs from frozen plan: {key}")
    for _, manifest, _ in loaded.values():
        for key in ["model_revision", "chat_template_sha256", "problem_set_sha256", "answer_grading"]:
            if manifest[key] != bm[key]:
                raise ValueError(f"mismatched evaluation: {key}")
        for key in ["seeds", "budgets", "batch_size", "max_tokens", "temperature", "top_p", "wording"]:
            if manifest["config"][key] != bm["config"][key]:
                raise ValueError(f"mismatched generation setting: {key}")
    base = loaded["base"][2]
    report = {"plan_sha256": digest(plan), "environment": environment(), "arms": {}}
    for name, (_, _, rows) in loaded.items():
        report["arms"][name] = {"results": summarize(rows), "paired_control": paired_control(rows)}
        if name != "base":
            report["arms"][name]["paired_differences"] = paired_difference(rows, base)
            report["arms"][name]["train_gate"] = train_gate(
                [r for r in rows if r["dataset"] == "sanity_train"],
                [r for r in base if r["dataset"] == "sanity_train"])
    passed = [name for name in ["sequence", "token"] if report["arms"][name]["train_gate"]["passed"]]
    report["selected_loss"] = passed[0] if passed else None
    report["stage_two_permitted"] = bool(passed)
    report["note"] = "A pass permits a larger SFT pilot only; no GRPO or release. Existing 100-problem holdout was not used."
    write_json(root/"comparison.json", report)
    print(json.dumps({"selected_loss": report["selected_loss"],
                      "gates": {n: report["arms"][n]["train_gate"] for n in ["token", "sequence"]}}, indent=2))
    return bool(passed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "run", "analyze"])
    ap.add_argument("--out-dir", default="results/sanity-2026-09-13")
    ap.add_argument("--data-dir", default="data/sanity-2026-09-13")
    args = ap.parse_args()
    root, data_root = Path(args.out_dir), Path(args.data_dir)
    if args.action == "prepare":
        if any(p.exists() and any(p.iterdir()) for p in [root, data_root]):
            ap.error("preparation directories must be new or empty")
        records = read_records("data/pilot/sft.jsonl")
        source = json.loads(Path("data/pilot/sft.manifest.json").read_text())
        if digest(records) != source["sft_sha256"]:
            ap.error("verified source dataset checksum mismatch")
        training, probe = select_ids(records)
        lookup = {r["problem_id"]: r for r in read_records("data/pilot/train_problems.jsonl")}
        dev = {r["problem_id"] for r in read_records("data/pilot/dev_problems.jsonl")}
        if set(training + probe) & dev:
            ap.error("diagnostic includes a held-out development problem")
        selected = [r for r in records if r["problem_id"] in training]
        if len(selected) != 72 or any({r["budget"] for r in selected if r["problem_id"] == i}
                                      != {512, 1024, 3600} for i in training):
            ap.error("expected 24 complete three-budget sets")
        problems = [{**lookup[i], "dataset": label} for label, ids in
                    [("sanity_train", training), ("sanity_probe", probe)] for i in ids]
        root.mkdir(parents=True, exist_ok=True)
        data_root.mkdir(parents=True, exist_ok=True)
        for path, rows in [(data_root/"sft.jsonl", selected), (root/"problems.jsonl", problems)]:
            with path.open("x") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(data_root/"sft.manifest.json", {**source, "parent_sft_sha256": source["sft_sha256"],
                   "sft_sha256": digest(selected), "n_problems": 24, "n_examples": 72, "selection_seed": 73})
        plan = {"schema_version": 1, "model": BASE, "revision": REVISION, "selection_seed": 73,
                "train_ids": training, "probe_ids": probe, "data": str(data_root/"sft.jsonl"),
                "data_sha256": digest(selected), "problem_sha256": digest(problems),
                "epochs": 20, "loss_type": "nll", "lr": 1e-4, "rank": 32, "seed": 42, "batch_size": 36,
                "budgets": [512, 1024, 3600], "max_tokens": 8192,
                "loss_modes": ["token", "sequence"], "checkpoint_root": "checkpoints/"+root.name,
                "control_report": "results/control-2026-09-10-regraded/comparison.json",
                "gate": "reference-control checks on training problems, plus >=75% accuracy at every budget",
                "selection": "prefer sequence if both pass; otherwise choose the sole passing arm",
                "probe": "12 separate verified-source problems; diagnostic only, not the 100-problem holdout",
                "stage_one_spending_limit_usd": 3, "environment": environment()}
        write_json(root/"plan.json", plan)
        print("Frozen 24 training problems / 72 examples and 12 diagnostic probe problems.")
        return
    if args.action == "analyze":
        raise SystemExit(0 if analyze(root) else 2)
    plan = json.loads((root/"plan.json").read_text())
    if digest(read_records(plan["data"])) != plan["data_sha256"] or digest(read_records(root/"problems.jsonl")) != plan["problem_sha256"]:
        ap.error("frozen data changed")
    def evaluate(name, adapter=None):
        command = [sys.executable, "eval_budget.py", "--model", plan["model"], "--revision", plan["revision"],
                   "--problem-file", str(root/"problems.jsonl"), "--budgets", *map(str, plan["budgets"]),
                   "--seeds", str(plan["seed"]), "--batch-size", str(plan["batch_size"]),
                   "--max-tokens", str(plan["max_tokens"]), "--out", str(root/(name+".json"))]
        if adapter:
            command += ["--lora", str(adapter)]
        subprocess.run(command, check=True)
    evaluate("base")
    for name in plan["loss_modes"]:
        output = Path(plan["checkpoint_root"])/name
        subprocess.run([sys.executable, "train_sft.py", "--data", plan["data"], "--control-report", plan["control_report"],
                        "--model", plan["model"], "--revision", plan["revision"], "--epochs", str(plan["epochs"]),
                        "--lr", str(plan["lr"]), "--rank", str(plan["rank"]), "--seed", str(plan["seed"]),
                        "--loss-weighting", name, "--save-strategy", "no", "--out-dir", str(output)], check=True)
        evaluate(name, output/"final")
    raise SystemExit(0 if analyze(root) else 2)


if __name__ == "__main__":
    main()

"""Frozen evaluation-only replication and decoding diagnostic; no training authorization."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from compare_pilot import load_evaluation
from compare_results import paired_difference
from eval_budget import summarize, paired_control
from experiment import digest, environment, read_records, write_json
from sanity_sft import train_gate

RUN = "decode-2026-09-14"
ROOT = Path("results") / RUN


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decision(stochastic_passes, greedy_passes):
    replicated = [name for name in ["token", "sequence"] if all(stochastic_passes[name])]
    return {"replicated_training_gate_arms": replicated,
            "greedy_training_gate_arms": [name for name, passed in greedy_passes.items() if passed],
            "recommendation": ("consider a separately approved generalization test" if replicated else
                "investigate decoding sensitivity; do not scale training" if any(greedy_passes.values()) else
                "pause this training recipe; preserve findings and consider the existing reference model"),
            "stage_two_permitted": False,
            "note": "Diagnostic only. Original seed-42 failure is unchanged; no automatic training or release."}


def analyze(root=ROOT):
    plan = json.loads((root / "plan.json").read_text())
    report = {"plan_sha256": digest(plan), "modes": {}, "environment": environment()}
    stochastic, greedy = {}, {}
    for mode, settings in plan["modes"].items():
        arms = {name: load_evaluation(root / f"{mode}-{name}.json") for name in plan["arms"]}
        bm = arms["base"][1]
        for name, (_, manifest, _) in arms.items():
            for key, value in {"seeds": settings["seeds"], "temperature": settings["temperature"],
                               "top_p": .95, "batch_size": 36, "max_tokens": 8192,
                               "budgets": [512, 1024, 3600], "wording": "exact"}.items():
                if manifest["config"][key] != value:
                    raise ValueError(f"{mode}/{name}: wrong {key}")
            for key in ["model_revision", "chat_template_sha256", "problem_set_sha256", "answer_grading", "token_measure"]:
                if manifest[key] != bm[key]:
                    raise ValueError(f"{mode}/{name}: mismatched {key}")
            if manifest["model_revision"] != plan["revision"]:
                raise ValueError("wrong base revision")
            if manifest["problem_set_sha256"] != plan["problem_sha256"]:
                raise ValueError("wrong problem set")
            if manifest.get("adapter_sha256") != plan["arms"][name].get("sha256"):
                raise ValueError("wrong adapter")
        base = arms["base"][2]
        report["modes"][mode] = {}
        for name, (_, _, rows) in arms.items():
            entry = {"results": summarize(rows), "paired_control": paired_control(rows)}
            if name != "base":
                entry["paired_differences"] = paired_difference(rows, base)
                entry["training_gates_by_seed"] = {str(seed): train_gate(
                    [r for r in rows if r["dataset"] == "sanity_train" and r["seed"] == seed],
                    [r for r in base if r["dataset"] == "sanity_train" and r["seed"] == seed])
                    for seed in settings["seeds"]}
                passed = [g["passed"] for g in entry["training_gates_by_seed"].values()]
                if mode == "stochastic":
                    stochastic[name] = passed
                else:
                    greedy[name] = all(passed)
            report["modes"][mode][name] = entry
    report["decision"] = decision(stochastic, greedy)
    write_json(root / "comparison.json", report)
    print(json.dumps(report["decision"], indent=2))


def prepare():
    previous = Path("results/sanity-2026-09-13")
    checkpoint = Path("checkpoints") / RUN
    if ROOT.exists() or checkpoint.exists():
        raise ValueError("diagnostic directories must be new")
    old = json.loads((previous / "plan.json").read_text())
    manifest = json.loads((previous / "artifact_hashes.json").read_text())
    arms = {"base": {}}
    for name in ["token", "sequence"]:
        source = Path("checkpoints/sanity-2026-09-13") / name / "final"
        for path in source.iterdir():
            if path.is_file() and sha(path) != manifest[str(path)]:
                raise ValueError(f"saved checkpoint changed: {path.name}")
        target = checkpoint / name / "final"
        shutil.copytree(source, target)
        arms[name] = {"path": str(target), "sha256": sha(target / "adapter_model.safetensors")}
    ROOT.mkdir(parents=True)
    shutil.copy2(previous / "problems.jsonl", ROOT / "problems.jsonl")
    write_json(ROOT / "plan.json", {"model": old["model"], "revision": old["revision"],
        "problem_sha256": digest(read_records(ROOT / "problems.jsonl")), "arms": arms,
        "modes": {"stochastic": {"temperature": .6, "seeds": [43, 44]},
                  "greedy": {"temperature": 0, "seeds": [42]}},
        "budgets": [512, 1024, 3600], "max_tokens": 8192, "batch_size": 36, "top_p": .95,
        "additional_spending_limit_usd": 1.5, "prior_stage_one_charges_usd": .519,
        "training": False, "stage_two_permitted": False,
        "decision_rule": "Replicated signal requires one arm to pass every existing training check separately at seeds 43 and 44. Greedy is secondary. Neither outcome unlocks stage two.",
        "stopping_rule": "If neither arm replicates across fresh seeds nor passes the greedy training diagnostic, recommend pausing this recipe, not another training run.",
        "scope": "Same 24 training / 12 filtered-source probes. Original 100-problem holdout untouched. No new training or changed hard caps.",
        "environment": environment()})


def run():
    plan = json.loads((ROOT / "plan.json").read_text())
    if digest(read_records(ROOT / "problems.jsonl")) != plan["problem_sha256"]:
        raise ValueError("frozen problem set changed")
    for arm in plan["arms"].values():
        if arm and sha(Path(arm["path"]) / "adapter_model.safetensors") != arm["sha256"]:
            raise ValueError("frozen adapter changed")
    for mode, settings in plan["modes"].items():
        for name, arm in plan["arms"].items():
            cmd = [sys.executable, "eval_budget.py", "--model", plan["model"], "--revision", plan["revision"],
                   "--problem-file", str(ROOT / "problems.jsonl"), "--budgets", "512", "1024", "3600",
                   "--max-tokens", "8192", "--batch-size", "36", "--top-p", ".95",
                   "--temperature", str(settings["temperature"]), "--seeds", *map(str, settings["seeds"]),
                   "--out", str(ROOT / f"{mode}-{name}.json")]
            if arm:
                cmd += ["--lora", arm["path"]]
            subprocess.run(cmd, check=True)
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.action]()

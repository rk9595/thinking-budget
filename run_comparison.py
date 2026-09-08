"""Freeze and run the control experiment; no training or publishing side effects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from eval_budget import select_problems
from experiment import artifact_path, digest, environment, read_records, resolve_model, write_json

BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEACHER = "l3lab/L1-Qwen-1.5B-Exact"
ADAPTER = "rk9595/thinking-budget-qwen1.5b-lcpo"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prepare-only", action="store_true", help="freeze problems/revisions without GPU work")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--adapter", help="use a local adapter instead of downloading run 3")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    if args.limit < 1:
        ap.error("limit must be positive")
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root/"comparison_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("protocol_version") != 2:
            ap.error("old comparison protocol; create a new plan directory")
        if plan["limit"] != args.limit or plan["seeds"] != args.seeds or plan["local_adapter"] != args.adapter:
            ap.error("existing plan has different settings; use a new directory")
    else:
        problems = select_problems(["math500"], args.limit, 17)
        revisions = {model: resolve_model(model)[1] for model in [BASE, TEACHER, ADAPTER]}
        problem_path = root/"problems.jsonl"
        with problem_path.open("x") as f:
            for p in problems:
                f.write(json.dumps(p, ensure_ascii=False)+"\n")
        plan = {"protocol_version": 2, "limit": args.limit, "seeds": args.seeds, "local_adapter": args.adapter,
                "budgets": [512, 1024, 3600], "max_tokens": 8192,
                "revisions": revisions, "problem_set_sha256": digest(problems),
                "environment": environment(),
                "arms": [{"name": "base_exact", "model": BASE, "wording": "exact"},
                         {"name": "run3_exact", "model": BASE, "wording": "exact", "adapter": True},
                         {"name": "base_l1", "model": BASE, "wording": "l1", "template_model": TEACHER},
                         {"name": "reference_l1", "model": TEACHER, "wording": "l1", "template_model": TEACHER}]}
        write_json(plan_path, plan)
    print(f"Plan frozen at {plan_path}; 3 models / 4 arms to control prompt wording.", flush=True)
    if digest(read_records(root/"problems.jsonl")) != plan["problem_set_sha256"]:
        raise ValueError("frozen problem file was modified")
    if args.prepare_only:
        return
    from huggingface_hub import snapshot_download
    adapter = args.adapter or str(Path(snapshot_download(
        ADAPTER, revision=plan["revisions"][ADAPTER], allow_patterns=["adapter/*"])) / "adapter")
    with (Path(adapter)/"adapter_model.safetensors").open("rb") as f:
        adapter_sha = hashlib.file_digest(f, "sha256").hexdigest()
    script_root = Path(__file__).resolve().parent
    for arm in plan["arms"]:
        output = root/(arm["name"]+".json")
        # Completed arms can be reused after a later arm fails. Partial files require a fresh path.
        if output.exists():
            previous = json.loads(output.read_text())
            manifest = json.loads(artifact_path(output, previous["manifest"]).read_text())
            cfg = manifest["config"]
            if (manifest["problem_set_sha256"] != plan["problem_set_sha256"]
                    or manifest["model_revision"] != plan["revisions"][arm["model"]]
                    or cfg["seeds"] != plan["seeds"] or cfg["wording"] != arm["wording"]
                    or cfg["budgets"] != plan["budgets"]
                    or cfg["max_tokens"] != plan["max_tokens"]
                    or bool(cfg["lora"]) != bool(arm.get("adapter"))
                    or (arm.get("adapter") and manifest.get("adapter_sha256") != adapter_sha)
                    or cfg.get("chat_template_model") != arm.get("template_model")
                    or (arm.get("template_model") and manifest.get("chat_template_revision") !=
                        plan["revisions"][arm["template_model"]])
                    or cfg["temperature"] != .6 or cfg["top_p"] != .95):
                raise ValueError(f"stale result: {output}")
            continue
        cmd = [sys.executable, str(script_root/"eval_budget.py"), "--model", arm["model"],
               "--revision", plan["revisions"][arm["model"]], "--wording", arm["wording"],
               "--problem-file", str(root/"problems.jsonl"), "--out", str(output),
               "--max-tokens", str(plan["max_tokens"]), "--budgets", *map(str, plan["budgets"]),
               "--seeds", *map(str, plan["seeds"])]
        if arm.get("adapter"):
            cmd += ["--lora", adapter]
        if arm.get("template_model"):
            cmd += ["--chat-template-model", arm["template_model"],
                    "--chat-template-revision", plan["revisions"][arm["template_model"]]]
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(script_root/"compare_results.py"),
                    "--run-dir", str(root)], check=True)


if __name__ == "__main__":
    main()

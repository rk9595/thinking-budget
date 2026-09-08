"""Prepare disjoint pilot problems, then filter saved teacher responses into SFT data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

from experiment import artifact_path, build_prompt, digest, read_records, write_json
from rewards import is_correct


def normalized(problem):
    return re.sub(r"\s+", " ", problem).strip().casefold()


def split_problems(rows, excluded, train_count, dev_count, seed):
    seen = {normalized(p) for p in excluded}
    clean = []
    for row in rows:
        key = normalized(row["problem"])
        if key in seen or not str(row["answer"]).strip():
            continue
        seen.add(key)
        clean.append({"dataset": "deepscaler_pilot", "problem_id": digest(key),
                      "problem": row["problem"], "answer": str(row["answer"])})
    random.Random(seed).shuffle(clean)
    if len(clean) < train_count + dev_count:
        raise ValueError("not enough unique problems after exclusions")
    return clean[:train_count], clean[train_count:train_count+dev_count]


def filter_teacher(samples, training, budgets, tolerance):
    by_id = {p["problem_id"]: p for p in training}
    accepted = defaultdict(dict)
    for row in samples:
        if row["problem_id"] not in by_id:
            raise ValueError("teacher output includes a problem outside the training split")
        gold = by_id[row["problem_id"]]
        budget = row["budget"]
        if row["problem"] != gold["problem"] or row["answer"] != gold["answer"]:
            raise ValueError("teacher problem/answer differs from frozen training split")
        if (budget not in budgets or row["finish_reason"] != "stop"
                or row["reasoning_tokens"] is None or not is_correct(row["text"], gold["answer"])
                or abs(row["total_tokens"]-budget) > tolerance*budget):
            continue
        previous = accepted[row["problem_id"]].get(budget)
        if previous is None or abs(row["total_tokens"]-budget) < abs(previous["total_tokens"]-budget):
            accepted[row["problem_id"]][budget] = row
    records = []
    # Keep complete budget sets so a problem is demonstrated at multiple lengths.
    for problem_id, variants in accepted.items():
        if set(variants) != set(budgets):
            continue
        for budget in budgets:
            row = variants[budget]
            records.append({"problem_id": problem_id, "budget": budget,
                            "prompt": [{"role": "user", "content": build_prompt(row["problem"], budget)}],
                            "completion": [{"role": "assistant", "content": row["text"]}],
                            "teacher_total_tokens": row["total_tokens"]})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("split")
    prepare.add_argument("--out-dir", default="data/pilot")
    prepare.add_argument("--train-count", type=int, default=500)
    prepare.add_argument("--dev-count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=29)
    filter_ap = sub.add_parser("filter")
    filter_ap.add_argument("--teacher-eval", required=True, help="completed eval_budget.py summary JSON")
    filter_ap.add_argument("--train-problems", default="data/pilot/train_problems.jsonl")
    filter_ap.add_argument("--out", default="data/pilot/sft.jsonl")
    filter_ap.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 3600])
    filter_ap.add_argument("--tolerance", type=float, default=.25)
    filter_ap.add_argument("--min-problems", type=int, default=100)
    args = ap.parse_args()
    if args.command == "split":
        from datasets import load_dataset
        from eval_budget import load_eval_set
        root = Path(args.out_dir)
        if root.exists() and any(root.iterdir()):
            ap.error("split directory must be new")
        if min(args.train_count, args.dev_count) < 1:
            ap.error("split counts must be positive")
        ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        excluded = [p for name in ["math500", "gsm8k"] for p, _ in load_eval_set(name)]
        train, dev = split_problems(ds, excluded, args.train_count, args.dev_count, args.seed)
        root.mkdir(parents=True, exist_ok=True)
        for name, records in [("train", train), ("dev", dev)]:
            with (root/(name+"_problems.jsonl")).open("x") as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
        write_json(root/"split_manifest.json", {"seed": args.seed,
                   "source_fingerprint": ds._fingerprint, "train_sha256": digest(train),
                   "dev_sha256": digest(dev), "train_count": len(train), "dev_count": len(dev),
                   "exclusions": "normalized exact matches against full MATH-500 and GSM8K test; not semantic decontamination"})
        print(f"Prepared {len(train)} training and {len(dev)} development problems")
    else:
        output = Path(args.out)
        if output.exists():
            ap.error("output already exists")
        if not 0 < args.tolerance < 1 or args.min_problems < 1:
            ap.error("tolerance must be in (0,1); min-problems must be positive")
        doc = json.loads(Path(args.teacher_eval).read_text())
        if doc["model"] != "l3lab/L1-Qwen-1.5B-Exact":
            ap.error("expected the reference L1-Exact teacher")
        records = filter_teacher(read_records(artifact_path(args.teacher_eval, doc["samples"])), read_records(args.train_problems),
                                 args.budgets, args.tolerance)
        count = len({r["problem_id"] for r in records})
        if count < args.min_problems:
            ap.error(f"only {count} complete correct budget sets; need {args.min_problems}. Inspect teacher outputs.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")
        write_json(output.with_suffix(".manifest.json"), {"teacher_eval": args.teacher_eval,
                   "teacher_manifest": json.loads(artifact_path(args.teacher_eval, doc["manifest"]).read_text()),
                   "sft_sha256": digest(records), "n_problems": count, "n_examples": len(records),
                   "budgets": args.budgets, "tolerance": args.tolerance})
        print(f"Accepted {count} problems / {len(records)} verified examples")


if __name__ == "__main__":
    main()

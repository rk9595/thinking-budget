"""Post-hoc error analysis of saved SFT diagnostics; never changes the frozen gate."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import statistics

from compare_pilot import load_evaluation
from experiment import write_json


def summarize_errors(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["budget"])].append(row)
    report = []
    for (dataset, budget), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: abs(r["total_tokens"] - budget), reverse=True)
        errors = [abs(r["total_tokens"] - budget) / budget for r in ordered]
        report.append({"dataset": dataset, "budget": budget, "n": len(group),
            "mean_absolute_relative_error": statistics.mean(errors),
            "median_absolute_relative_error": statistics.median(errors),
            "largest_two_share_of_absolute_error": sum(errors[:2]) / sum(errors) if sum(errors) else 0,
            "closed_think_count": sum("</think>" in r["text"] for r in group),
            "clipped_count": sum(r["finish_reason"] == "length" for r in group),
            "outliers": [{"problem_id": r["problem_id"], "problem": r["problem"],
                          "total_tokens": r["total_tokens"], "correct": r["correct"],
                          "finish_reason": r["finish_reason"], "ending_excerpt": r["text"][-400:]}
                         for r in ordered[:3]]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/sanity-2026-09-13")
    parser.add_argument("--out", default="results/sanity-2026-09-13/error_audit.json")
    args = parser.parse_args()
    root, out = Path(args.run_dir), Path(args.out)
    if out.exists():
        parser.error("audit already exists; use a new output path")
    report = {"note": "Post-hoc diagnostic only. Outliers remain in all metrics and original gates remain unchanged.",
              "arms": {}, "source_sha256": {}}
    for name in ["base", "token", "sequence"]:
        _, _, rows = load_evaluation(root / f"{name}.json")
        report["arms"][name] = summarize_errors(rows)
        for suffix in [".json", ".manifest.json", ".samples.jsonl"]:
            path = root / (name + suffix)
            report["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out, report)


if __name__ == "__main__":
    main()

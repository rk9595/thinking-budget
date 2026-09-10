"""Re-grade saved responses with prefill-aware correctness, preserving the originals."""
import argparse
import json
from pathlib import Path

from compare_results import validate_samples
from eval_budget import paired_control, summarize
from experiment import artifact_path, digest, environment, read_records, write_json
from rewards import is_correct


def regrade(source, output):
    source, output = Path(source), Path(output)
    raw_path = output.with_suffix(".samples.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    if any(p.exists() for p in [output, raw_path, manifest_path]):
        raise ValueError("regrade outputs must be new; originals are never overwritten")
    doc = json.loads(source.read_text())
    manifest = json.loads(artifact_path(source, doc["manifest"]).read_text())
    original = read_records(artifact_path(source, doc["samples"]))
    validate_samples(original, manifest)
    rows = [{**r, "legacy_correct": r["correct"],
             "correct": is_correct(r["text"], r["answer"],
                 prefilled_think=r["rendered_prompt"].rstrip().endswith("<think>"))} for r in original]
    changed = sum(r["correct"] != r["legacy_correct"] for r in rows)
    manifest["answer_grading"] = "prefilled_think_aware_v2"
    manifest["regrade"] = {"source_eval": str(source), "source_samples_sha256": digest(original),
                          "changed_grades": changed, "environment": environment()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(manifest_path, manifest)
    write_json(output, {**doc, "manifest": manifest_path.name, "samples": raw_path.name,
                       "results": summarize(rows), "paired_control": paired_control(rows)})
    print(f"{source.name}: regraded {len(rows)} saved responses; {changed} grades changed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    regrade(args.input, args.out)


if __name__ == "__main__":
    main()

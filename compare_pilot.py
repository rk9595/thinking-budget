"""Compare a held-out SFT pilot with its identically configured unadapted base."""
import argparse
import json
from pathlib import Path

from compare_results import paired_difference, reference_gate, validate_samples
from eval_budget import paired_control, summarize
from experiment import artifact_path, read_records, write_json


def pilot_gate(candidate, base):
    # Reuse the predeclared reference-control diagnostic, then add retention.
    checks = dict(reference_gate(candidate, base)["checks"])
    candidate_summary, base_summary = summarize(candidate), summarize(base)
    highest = max(r["budget"] for r in base_summary)
    base_accuracy = {r["dataset"]: r["accuracy"] for r in base_summary if r["budget"] == highest}
    checks["high_budget_accuracy_loss_at_most_10_points"] = all(
        r["accuracy"] >= base_accuracy[r["dataset"]] - .10 - 1e-12
        for r in candidate_summary if r["budget"] == highest)
    return {"ready_for_grpo_pilot": all(checks.values()), "checks": checks,
            "note": "Development screening only, using point estimates. Inspect uncertainty and raw traces; not release approval."}


def load_evaluation(path):
    doc = json.loads(Path(path).read_text())
    manifest = json.loads(artifact_path(path, doc["manifest"]).read_text())
    samples = read_records(artifact_path(path, doc["samples"]))
    validate_samples(samples, manifest)
    return doc, manifest, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        ap.error("output already exists; use a new report path")
    bdoc, bm, base = load_evaluation(args.base)
    cdoc, cm, candidate = load_evaluation(args.candidate)
    if bm.get("answer_grading", "legacy_v1") != cm.get("answer_grading", "legacy_v1"):
        ap.error("mismatched answer-grading versions")
    if bdoc["lora"] is not None or cdoc["lora"] is None or bdoc["model"] != cdoc["model"]:
        ap.error("expected an unadapted base and a LoRA on the same base model")
    for key in ["problem_set_sha256", "token_measure", "chat_template_sha256", "model_revision"]:
        if bm[key] != cm[key]:
            ap.error(f"mismatched evaluation field: {key}")
    for key in ["seeds", "budgets", "max_tokens", "temperature", "top_p", "batch_size", "wording"]:
        if bm["config"][key] != cm["config"][key]:
            ap.error(f"mismatched sampling setting: {key}")
    result = {"schema_version": 1, "base_eval": args.base, "candidate_eval": args.candidate,
              "base": {"results": summarize(base), "paired_control": paired_control(base)},
              "candidate": {"results": summarize(candidate), "paired_control": paired_control(candidate)},
              "paired_differences": paired_difference(candidate, base),
              "pilot_gate": pilot_gate(candidate, base)}
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not result["pilot_gate"]["ready_for_grpo_pilot"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

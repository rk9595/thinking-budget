"""Compare frozen evaluation arms and test whether the reference control is detected."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from experiment import artifact_path, read_records, write_json


def bootstrap_mean(values, seed=123, repeats=1000):
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [estimates[int(.025*repeats)], estimates[int(.975*repeats)]]


def validate_samples(samples, manifest):
    expected = {(p["dataset"], p["problem_id"], b, s)
                for p in manifest["problems"]
                for b in manifest["config"]["budgets"] for s in manifest["config"]["seeds"]}
    actual = {(r["dataset"], r["problem_id"], r["budget"], r["seed"]) for r in samples}
    if actual != expected or len(samples) != len(expected):
        raise ValueError("raw samples do not cover the complete frozen evaluation grid")


def paired_difference(candidate, baseline):
    def index(rows):
        result = {(r["dataset"], r["problem_id"], r["budget"], r["seed"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate responses in evaluation")
        return result
    c, b = index(candidate), index(baseline)
    if not c or c.keys() != b.keys():
        raise ValueError("comparisons require identical problems, budgets, and sampling seeds")
    grouped = defaultdict(lambda: defaultdict(list))
    for key, row in c.items():
        dataset, problem, budget, _ = key
        for metric in ("correct", "total_tokens"):
            grouped[(dataset, budget, metric)][problem].append(row[metric]-b[key][metric])
    result = []
    for (dataset, budget, metric), problems in sorted(grouped.items()):
        # Resample problems, not repeated seeds from the same problem.
        values = [statistics.mean(v) for v in problems.values()]
        result.append({"dataset": dataset, "budget": budget, "metric": metric,
                       "n_problems": len(values), "mean_delta": statistics.mean(values),
                       "bootstrap_95pct_ci": bootstrap_mean(values)})
    return result


def reference_gate(reference, base):
    """A pilot gate for detecting the known control, not a model release criterion."""
    from eval_budget import paired_control, summarize
    paired_difference(reference, base)  # Validate the full matched sample grid first.
    summaries = summarize(reference)
    control = paired_control(reference)
    base_control = paired_control(base)
    checks = {
        "length_error_under_35pct_each_budget": all(r["mean_relative_error"] <= .35 for r in summaries),
        "truncation_under_10pct_each_budget": all(r["truncation_rate"] < .1 for r in summaries),
        "positive_response_on_75pct_of_pairs": bool(control) and all(
            r["positive_endpoint_fraction"] >= .75 for r in control.values()),
        "slope_between_half_and_one_and_half": bool(control) and all(
            .5 <= r["mean_endpoint_slope"] <= 1.5 for r in control.values()),
        "slope_exceeds_base_by_point_three": bool(control) and all(
            r["mean_endpoint_slope"] - base_control[name]["mean_endpoint_slope"] >= .3
            for name, r in control.items()),
        "some_correct_answers_each_budget": all(r["accuracy"] >= .2 for r in summaries),
    }
    return {"reference_control_detected": all(checks.values()), "checks": checks,
            "note": "Development diagnostic only; thresholds fixed before generation. Not release approval."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    names = ["base_exact", "run3_exact", "base_l1", "reference_l1"]
    docs = {name: json.loads((root/(name+".json")).read_text()) for name in names}
    manifests = {name: json.loads(artifact_path(root/(name+".json"), doc["manifest"]).read_text())
                 for name, doc in docs.items()}
    signatures = {tuple(json.dumps(m["config"][k], sort_keys=True) for k in
                        ["seeds", "budgets", "max_tokens", "temperature", "top_p"])
                  + (m["problem_set_sha256"], m["token_measure"]) for m in manifests.values()}
    if len(signatures) != 1:
        raise ValueError("evaluation settings or problem sets do not match")
    for a, b in [("base_exact", "run3_exact"), ("base_l1", "reference_l1")]:
        if (manifests[a]["chat_template_sha256"] != manifests[b]["chat_template_sha256"]
                or manifests[a]["config"]["wording"] != manifests[b]["config"]["wording"]):
            raise ValueError("paired models must use the same wording and chat template")
    samples = {name: read_records(artifact_path(root/(name+".json"), doc["samples"]))
               for name, doc in docs.items()}
    from eval_budget import summarize, paired_control
    for name, rows in samples.items():
        validate_samples(rows, manifests[name])
        docs[name]["results"] = summarize(rows)
        docs[name]["paired_control"] = paired_control(rows)
    result = {"schema_version": 1, "arms": {name: {"results": d["results"],
              "paired_control": d["paired_control"]} for name, d in docs.items()},
              "run3_vs_base": paired_difference(samples["run3_exact"], samples["base_exact"]),
              "reference_vs_base": paired_difference(samples["reference_l1"], samples["base_l1"]),
              "reference_gate": reference_gate(samples["reference_l1"], samples["base_l1"])}
    write_json(root/"comparison.json", result)
    print("model | budget | accuracy | tokens | relative error | truncation")
    for name, doc in docs.items():
        for row in doc["results"]:
            print(f"{name} | {row['budget']} | {row['accuracy']:.1%} | {row['mean_tokens']:.0f} | "
                  f"{row['mean_relative_error']:.1%} | {row['truncation_rate']:.1%}")
    print(json.dumps(result["reference_gate"], indent=2))
    if not result["reference_gate"]["reference_control_detected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

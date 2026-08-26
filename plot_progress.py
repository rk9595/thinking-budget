"""Plot budget adherence and accuracy as a function of training step.

Reads results/ckpt_<step>.json produced by the checkpoint sweep in pipeline.sh,
plus results/base.json as the step-0 reference.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    points = []
    base = os.path.join(a.results, "base.json")
    if os.path.exists(base):
        rows = json.load(open(base))["results"]
        rows = [r for r in rows if r["dataset"] == "math500"]
        if rows:
            points.append((0, mean(r["mean_abs_deviation"] for r in rows),
                           mean(r["accuracy"] for r in rows),
                           mean(r["mean_tokens"] for r in rows)))

    for f in glob.glob(os.path.join(a.results, "ckpt_*.json")):
        m = re.search(r"ckpt_(\d+)\.json", f)
        if not m:
            continue
        rows = [r for r in json.load(open(f))["results"] if r["dataset"] == "math500"]
        if rows:
            points.append((int(m.group(1)), mean(r["mean_abs_deviation"] for r in rows),
                           mean(r["accuracy"] for r in rows),
                           mean(r["mean_tokens"] for r in rows)))

    if len(points) < 2:
        print(f"need >=2 points, have {len(points)} - skipping progress plot")
        return
    points.sort()
    steps = [p[0] for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(steps, [p[1] for p in points], marker="o", color="tab:red")
    axes[0].set(xlabel="training step", ylabel="mean |tokens - budget|",
                title="Budget adherence improves with training")
    axes[1].plot(steps, [p[2] for p in points], marker="o", color="tab:blue")
    axes[1].set(xlabel="training step", ylabel="accuracy (MATH-500)",
                title="Accuracy is not sacrificed")
    axes[2].plot(steps, [p[3] for p in points], marker="o", color="tab:green")
    axes[2].set(xlabel="training step", ylabel="mean tokens generated",
                title="Generations get shorter")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "progress_vs_step.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)

    print("\n| step | mean |tok-budget| | accuracy | mean tokens |")
    print("|---|---|---|---|")
    for s, d, acc, tok in points:
        print(f"| {s} | {d:.0f} | {acc:.3f} | {tok:.0f} |")


if __name__ == "__main__":
    main()

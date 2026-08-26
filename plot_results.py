import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, label):
    with open(path) as f:
        d = json.load(f)
    return label, d["results"], d.get("budget_wording", "max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/base.json")
    ap.add_argument("--trained", default="results/trained.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    loaded = [load(args.base, "base"), load(args.trained, "trained")]
    wording = loaded[1][2]
    runs = [(label, rows) for label, rows, _ in loaded]
    datasets = sorted({r["dataset"] for _, rows in runs for r in rows})

    for metric, ylabel, fname in [
        ("accuracy", "accuracy", "accuracy_vs_budget.png"),
        ("mean_tokens", "mean tokens generated", "tokens_vs_budget.png"),
        ("mean_abs_deviation", "mean |tokens - budget|", "adherence_vs_budget.png"),
        ("overshoot_rate", "fraction over budget", "overshoot_vs_budget.png"),
        ("mean_overshoot", "mean max(0, tokens - budget)", "mean_overshoot_vs_budget.png"),
    ]:
        # runs captured before the max-variant switch have no overshoot columns
        if not any(metric in r for _, rows in runs for r in rows):
            continue
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            for label, rows in runs:
                pts = sorted((r["budget"], r[metric]) for r in rows if r["dataset"] == ds and metric in r)
                if pts:
                    ax.plot(*zip(*pts), marker="o", label=label)
            if metric == "mean_tokens":
                lim = [r["budget"] for _, rows in runs for r in rows]
                ax.plot([min(lim), max(lim)], [min(lim), max(lim)], "k--", alpha=0.4, label="budget (y=x)")
            ax.set_title(ds)
            ax.set_xlabel("token budget")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, fname)
        fig.savefig(path, dpi=150)
        print("wrote", path)

    # Exact targets the budget from both sides, so |tokens - budget| is the
    # headline; Max only cares about the ceiling, so overshoot is.
    key = "mean_abs_deviation" if wording == "exact" else "mean_overshoot"
    if not all(key in r for _, rows in runs for r in rows):
        key = "mean_abs_deviation"
    print(f"\n| dataset | budget | acc base | acc trained | {key} base | {key} trained |")
    print("|---|---|---|---|---|---|")
    base = {(r["dataset"], r["budget"]): r for r in runs[0][1]}
    for r in sorted(runs[1][1], key=lambda x: (x["dataset"], x["budget"])):
        b = base.get((r["dataset"], r["budget"]))
        if b:
            print(
                f"| {r['dataset']} | {r['budget']} | {b['accuracy']:.3f} | {r['accuracy']:.3f} "
                f"| {b[key]:.0f} | {r[key]:.0f} |"
            )


if __name__ == "__main__":
    main()

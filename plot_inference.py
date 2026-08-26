"""Turn inference_profile.json + a gpu_monitor CSV into charts for the write-up."""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_profile(path, out_dir):
    d = json.load(open(path))
    rows = d["rows"]
    budgets = sorted({r["budget"] for r in rows})
    concs = sorted({r["concurrency"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for b in budgets:
        pts = sorted((r["concurrency"], r["output_tok_per_s"]) for r in rows if r["budget"] == b)
        axes[0].plot(*zip(*pts), marker="o", label=f"budget {b}")
    axes[0].set(xlabel="concurrent requests", ylabel="output tokens/sec",
                title="Throughput scales with batching")
    axes[0].set_xscale("log", base=2)

    for c in concs:
        pts = sorted((r["budget"], r["latency_per_req_s"]) for r in rows if r["concurrency"] == c)
        axes[1].plot(*zip(*pts), marker="o", label=f"conc {c}")
    axes[1].set(xlabel="thinking budget (tokens)", ylabel="seconds per request",
                title="Budget is a latency dial")

    for c in concs:
        pts = sorted((r["budget"], r["cost_per_1k_req_usd"]) for r in rows if r["concurrency"] == c)
        axes[2].plot(*zip(*pts), marker="o", label=f"conc {c}")
    axes[2].set(xlabel="thinking budget (tokens)", ylabel="$ per 1000 requests",
                title=f"Serving cost @ ${d['dph']}/hr")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "inference_profile.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def plot_gpu(path, out_dir, name):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return
    t = [float(r["t"]) / 60 for r in rows]
    util = [float(r["utilization.gpu"]) for r in rows]
    mem = [float(r["memory.used"]) / 1024 for r in rows]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, util, lw=0.8, label="GPU util %")
    ax.set(xlabel="minutes", ylabel="GPU utilization (%)", ylim=(0, 105),
           title=f"GPU utilization during {name}")
    ax2 = ax.twinx()
    ax2.plot(t, mem, lw=0.8, color="tab:orange", alpha=0.6, label="VRAM GiB")
    ax2.set_ylabel("VRAM (GiB)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"gpu_{name}.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="results/inference_profile.json")
    ap.add_argument("--gpu", nargs="*", default=[])
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(a.profile):
        plot_profile(a.profile, a.out_dir)
    for g in a.gpu:
        if os.path.exists(g):
            plot_gpu(g, a.out_dir, os.path.basename(g).replace("gpu_", "").replace(".csv", ""))


if __name__ == "__main__":
    main()

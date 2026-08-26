"""Sample GPU telemetry to CSV while something else runs. Also summarizes a finished log.

  python gpu_monitor.py --out results/gpu_train.csv &      # sampler
  python gpu_monitor.py --summarize results/gpu_train.csv   # report
"""
import argparse
import csv
import subprocess
import sys
import time

FIELDS = ["utilization.gpu", "utilization.memory", "memory.used", "memory.total",
          "power.draw", "power.limit", "clocks.sm", "temperature.gpu"]


def sample():
    q = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    return [p.strip() for p in q.split(",")]


def run(out, interval):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + FIELDS)
        t0 = time.time()
        try:
            while True:
                w.writerow([round(time.time() - t0, 2)] + sample())
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty log")
        return
    util = [float(r["utilization.gpu"]) for r in rows]
    mem = [float(r["memory.used"]) for r in rows]
    pw = [float(r["power.draw"]) for r in rows]
    total = float(rows[0]["memory.total"])
    limit = float(rows[0]["power.limit"])
    dur = float(rows[-1]["t"])

    print(f"samples={len(rows)}  duration={dur/60:.1f} min")
    print(f"GPU util   mean={sum(util)/len(util):5.1f}%  p50={pct(util,.5):5.1f}%  p95={pct(util,.95):5.1f}%")
    print(f"           idle (<5%) {100*sum(u<5 for u in util)/len(util):.1f}% of samples")
    print(f"VRAM       mean={sum(mem)/len(mem)/1024:5.1f} GiB  peak={max(mem)/1024:5.1f} / {total/1024:.0f} GiB "
          f"({100*max(mem)/total:.0f}% peak)")
    print(f"Power      mean={sum(pw)/len(pw):5.1f} W  peak={max(pw):5.1f} / {limit:.0f} W "
          f"({100*(sum(pw)/len(pw))/limit:.0f}% of limit)")
    energy_kwh = (sum(pw) / len(pw)) * (dur / 3600) / 1000
    print(f"Energy     ~{energy_kwh:.3f} kWh over the window")
    if sum(util) / len(util) < 60:
        print("\n  Mean utilization under 60% means the GPU is stalling - usually data loading,\n"
              "  reward computation on CPU, or too-small a rollout batch.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gpu.csv")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--summarize")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
    else:
        run(a.out, a.interval)


if __name__ == "__main__":
    main()

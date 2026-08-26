"""Measure what the thinking budget actually buys you at serving time.

Sweeps (thinking budget x concurrency) through vLLM and reports throughput, latency,
GPU utilization and $/1k-requests. Optionally compares against HF generate to show
what continuous batching + paged KV are worth.

  python profile_inference.py --lora checkpoints/lcpo-max/checkpoint-300 \
      --dph 0.95 --out results/inference_profile.json
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time

INSTR = "Let's think step by step and output the final answer within \\boxed{}."


class GpuSampler(threading.Thread):
    """Polls nvidia-smi in the background so we can attribute utilization to a phase."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.samples, self._stop = interval, [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True).stdout.strip()
                u, m = (p.strip() for p in out.split(","))
                self.samples.append((float(u), float(m)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=3)
        if not self.samples:
            return {"gpu_util_mean": None, "vram_peak_gib": None}
        return {
            "gpu_util_mean": round(statistics.mean(u for u, _ in self.samples), 1),
            "vram_peak_gib": round(max(m for _, m in self.samples) / 1024, 2),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 128])
    ap.add_argument("--dph", type=float, default=0.95, help="instance $/hr, for cost math")
    ap.add_argument("--compare-hf", action="store_true", help="also time HF generate at bs=8")
    ap.add_argument("--out", default="results/inference_profile.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    problems = [ex["problem"] for ex in load_dataset("HuggingFaceH4/MATH-500", split="test")]

    def build(n, budget):
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": f"{problems[i % len(problems)]}\n\n{INSTR} "
                                             f"Think for maximum {budget} tokens."}],
                tokenize=False, add_generation_prompt=True)
            for i in range(n)
        ]

    llm = LLM(model=args.model, enable_lora=bool(args.lora), max_lora_rank=32,
              gpu_memory_utilization=0.90)
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, args.lora)

    rows = []
    for budget in args.budgets:
        for conc in args.concurrency:
            prompts = build(conc, budget)
            # cap generation at 2x budget so a non-compliant model cannot run away
            sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=budget * 2)
            sampler = GpuSampler()
            sampler.start()
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, lora_request=lora_req)
            dt = time.perf_counter() - t0
            gpu = sampler.stop()

            out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
            in_toks = sum(len(o.prompt_token_ids) for o in outs)
            row = {
                "budget": budget, "concurrency": conc,
                "wall_s": round(dt, 2),
                "output_tokens": out_toks,
                "output_tok_per_s": round(out_toks / dt, 1),
                "total_tok_per_s": round((out_toks + in_toks) / dt, 1),
                "mean_out_tokens": round(out_toks / conc, 1),
                "latency_per_req_s": round(dt / conc, 3),
                "req_per_s": round(conc / dt, 3),
                "cost_per_1k_req_usd": round((dt / conc) * 1000 / 3600 * args.dph, 4),
                **gpu,
            }
            rows.append(row)
            print(f"budget={budget:5d} conc={conc:4d} "
                  f"{row['output_tok_per_s']:8.1f} out-tok/s  "
                  f"{row['latency_per_req_s']:7.3f} s/req  "
                  f"util={row['gpu_util_mean']}%  "
                  f"${row['cost_per_1k_req_usd']}/1k-req")

    result = {"model": args.model, "lora": args.lora, "dph": args.dph, "rows": rows}

    if args.compare_hf:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
        tok.padding_side = "left"
        enc = tok(build(8, 512), return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=512, do_sample=True, temperature=0.6)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = int((o.shape[1] - enc["input_ids"].shape[1]) * o.shape[0])
        hf_tps = gen / dt
        vllm_tps = next((r["output_tok_per_s"] for r in rows
                         if r["budget"] == 512 and r["concurrency"] == 8), None)
        result["hf_baseline"] = {"output_tok_per_s": round(hf_tps, 1),
                                 "vllm_output_tok_per_s": vllm_tps,
                                 "speedup": round(vllm_tps / hf_tps, 2) if vllm_tps else None}
        print(f"\nHF generate bs=8: {hf_tps:.1f} tok/s vs vLLM {vllm_tps} tok/s "
              f"-> {result['hf_baseline']['speedup']}x")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()

"""Seeded budget evaluation with durable per-response evidence (CUDA/vLLM)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from experiment import build_prompt, digest, environment, read_records, resolve_model, write_json
from rewards import is_correct


def load_eval_set(name):
    from datasets import load_dataset
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [(ex["problem"], str(ex["answer"])) for ex in ds]
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [(ex["question"], ex["answer"].split("####")[-1].strip()) for ex in ds]
    if name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [(ex["problem"], str(ex["answer"])) for ex in ds]
    raise ValueError(f"unknown dataset {name}")


def select_problems(names, limit, seed):
    records = []
    for name in names:
        problems = load_eval_set(name)
        indices = list(range(len(problems)))
        random.Random(seed).shuffle(indices)
        for index in indices[:limit]:
            problem, answer = problems[index]
            records.append({"dataset": name, "problem_id": digest([name, problem]),
                            "source_index": index, "problem": problem, "answer": answer})
    return records


def reasoning_counts(ids, tokenizer):
    """Count original token IDs; missing delimiters are unknown, not zero reasoning."""
    close = tokenizer.encode("</think>", add_special_tokens=False)
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    for i in range(len(ids) - len(close) + 1):
        if close and list(ids[i:i + len(close)]) == close:
            start = len(opening) if opening and list(ids[:len(opening)]) == opening else 0
            return max(0, i - start), len(ids) - i - len(close)
    return None, None


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["budget"])].append(record)
    results = []
    for (dataset, budget), rows in sorted(grouped.items()):
        lengths = [r["total_tokens"] for r in rows]
        overs = [max(0, n - budget) for n in lengths]
        reasoning = [r["reasoning_tokens"] for r in rows if r["reasoning_tokens"] is not None]
        results.append({"dataset": dataset, "budget": budget, "n": len(rows),
                        "n_problems": len({r["problem_id"] for r in rows}),
                        "accuracy": statistics.mean(r["correct"] for r in rows),
                        "mean_tokens": statistics.mean(lengths),
                        "median_tokens": statistics.median(lengths),
                        "mean_abs_deviation": statistics.mean(abs(n-budget) for n in lengths),
                        "mean_relative_error": statistics.mean(abs(n-budget)/budget for n in lengths),
                        "within_20pct": statistics.mean(abs(n-budget) <= .2*budget for n in lengths),
                        "overshoot_rate": statistics.mean(n > 0 for n in overs),
                        "mean_overshoot": statistics.mean(overs),
                        "truncation_rate": statistics.mean(r["finish_reason"] == "length" for r in rows),
                        "think_closed_rate": len(reasoning)/len(rows),
                        "mean_reasoning_tokens": statistics.mean(reasoning) if reasoning else None})
    return results


def paired_control(records):
    groups = defaultdict(dict)
    for row in records:
        groups[(row["dataset"], row["problem_id"], row["seed"])][row["budget"]] = row["total_tokens"]
    by_dataset = defaultdict(list)
    for (dataset, _, _), lengths in groups.items():
        budgets = sorted(lengths)
        if len(budgets) < 2:
            continue
        low, high = budgets[0], budgets[-1]
        by_dataset[dataset].append({
            "increases": lengths[high] > lengths[low],
            "monotonic": all(lengths[b] >= lengths[a] for a, b in zip(budgets, budgets[1:])),
            "slope": (lengths[high]-lengths[low])/(high-low)})
    return {name: {"n_pairs": len(rows),
                   "positive_endpoint_fraction": statistics.mean(r["increases"] for r in rows),
                   "monotonic_fraction": statistics.mean(r["monotonic"] for r in rows),
                   "mean_endpoint_slope": statistics.mean(r["slope"] for r in rows)}
            for name, rows in by_dataset.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--revision")
    ap.add_argument("--chat-template-model", help="use this model's template for a matched control")
    ap.add_argument("--chat-template-revision")
    ap.add_argument("--lora", help="local adapter directory")
    ap.add_argument("--datasets", nargs="+", default=["math500", "gsm8k"])
    ap.add_argument("--problem-file", help="frozen JSONL problem set, used in full")
    ap.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 3600])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--selection-seed", type=int, default=17)
    ap.add_argument("--wording", choices=["exact", "max", "l1"], default="exact")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=.6)
    ap.add_argument("--top-p", type=float, default=.95)
    ap.add_argument("--gpu-memory", type=float, default=.85)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()
    if (args.limit < 1 or args.batch_size < 1 or min(args.budgets) < 1
            or args.max_tokens <= max(args.budgets)):
        ap.error("positive limits/budgets required; max-tokens must exceed every budget")
    if len(set(args.budgets)) != len(args.budgets) or len(set(args.seeds)) != len(args.seeds):
        ap.error("duplicate budgets/seeds are not allowed")
    out = Path(args.out)
    raw_path = out.with_suffix(".samples.jsonl")
    manifest_path = out.with_suffix(".manifest.json")
    if any(p.exists() for p in [out, raw_path, manifest_path]):
        ap.error("output already exists; choose a new output path to preserve evidence")
    problems = (read_records(args.problem_file) if args.problem_file
                else select_problems(args.datasets, args.limit, args.selection_seed))
    if not problems or len({p["problem_id"] for p in problems}) != len(problems):
        ap.error("problem set must be nonempty with unique problem IDs")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model, revision = resolve_model(args.model, args.revision)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    template_revision = revision
    if args.chat_template_model:
        template_model, template_revision = resolve_model(args.chat_template_model, args.chat_template_revision)
        tok.chat_template = AutoTokenizer.from_pretrained(template_model, revision=template_revision).chat_template
    rank = 32
    if args.lora:
        config = json.loads((Path(args.lora)/"adapter_config.json").read_text())
        rank = max(8, config["r"])
        if config["base_model_name_or_path"] != args.model:
            raise ValueError("adapter base does not match --model")
    manifest = {"schema_version": 2, "config": vars(args), "model_revision": revision,
                "problem_set_sha256": digest(problems), "problems": problems,
                "token_measure": "all generated token IDs, including EOS if returned",
                "chat_template_sha256": digest(tok.chat_template),
                "chat_template_revision": template_revision, "environment": environment()}
    if args.lora:
        import hashlib
        with (Path(args.lora)/"adapter_model.safetensors").open("rb") as f:
            manifest["adapter_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    write_json(manifest_path, manifest)
    llm = LLM(model=model, revision=revision, tokenizer_revision=revision,
              enable_lora=bool(args.lora), max_lora_rank=rank, dtype="bfloat16",
              seed=args.seeds[0], max_model_len=args.max_tokens + 2048,
              gpu_memory_utilization=args.gpu_memory)
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, args.lora)
    records = []
    with raw_path.open("x") as raw:
        for seed in args.seeds:
            for budget in args.budgets:
                for start in range(0, len(problems), args.batch_size):
                    batch = problems[start:start + args.batch_size]
                    prompts = [tok.apply_chat_template(
                        [{"role": "user", "content": build_prompt(p["problem"], budget, args.wording)}],
                        tokenize=False, add_generation_prompt=True) for p in batch]
                    params = [SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens,
                              seed=(seed + int(p["problem_id"][:8], 16)) % (2**31)) for p in batch]
                    outputs = llm.generate(prompts, params, lora_request=lora_request)
                    if len(outputs) != len(batch):
                        raise RuntimeError("generation count does not match problem count")
                    for p, prompt, generated in zip(batch, prompts, outputs):
                        result = generated.outputs[0]
                        reasoning, answer = reasoning_counts(result.token_ids, tok)
                        row = {**p, "budget": budget, "seed": seed, "rendered_prompt": prompt,
                               "text": result.text, "token_ids": list(result.token_ids),
                               "total_tokens": len(result.token_ids), "reasoning_tokens": reasoning,
                               "answer_tokens": answer, "finish_reason": result.finish_reason,
                               "stop_reason": result.stop_reason,
                               "correct": is_correct(result.text, p["answer"])}
                        records.append(row)
                        raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw.flush()
                print(f"completed budget={budget} seed={seed}", flush=True)
    write_json(out, {"schema_version": 2, "model": model, "lora": args.lora,
                    "budget_wording": args.wording, "manifest": manifest_path.name,
                    "samples": raw_path.name, "results": summarize(records),
                    "paired_control": paired_control(records)})
    print(f"saved {len(records)} responses to {raw_path}")


if __name__ == "__main__":
    main()

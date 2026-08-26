import argparse
import json
import os

from datasets import load_dataset

from rewards import LENGTH_REWARD_VARIANT, budget_instruction, is_correct

INSTR = "Let's think step by step and output the final answer within \\boxed{}."


def load_eval_set(name):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--lora", default=None, help="path to LoRA adapter checkpoint")
    ap.add_argument("--datasets", nargs="+", default=["math500", "gsm8k"])
    # A 3600 budget is above the base model's natural ~2500 tokens, so nothing
    # ever overshoots it; sweep where the ceiling actually binds.
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--limit", type=int, default=200, help="problems per dataset")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, enable_lora=bool(args.lora), max_lora_rank=32)
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, args.lora)
    sampling = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens)

    results = []
    for ds_name in args.datasets:
        problems = load_eval_set(ds_name)[: args.limit]
        for budget in args.budgets:
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": f"{p}\n\n{INSTR} {budget_instruction(budget)}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p, _ in problems
            ]
            outs = llm.generate(prompts, sampling, lora_request=lora_request)
            n_correct, deviations, overshoots, lengths = 0, [], [], []
            for out, (_, gold) in zip(outs, problems):
                text = out.outputs[0].text
                n_tokens = len(out.outputs[0].token_ids)
                lengths.append(n_tokens)
                deviations.append(abs(n_tokens - budget))
                overshoots.append(max(0, n_tokens - budget))
                if is_correct(text, gold):
                    n_correct += 1
            row = {
                "dataset": ds_name,
                "budget": budget,
                "n": len(problems),
                "accuracy": n_correct / len(problems),
                "mean_tokens": sum(lengths) / len(lengths),
                "mean_abs_deviation": sum(deviations) / len(deviations),
                "overshoot_rate": sum(1 for o in overshoots if o > 0) / len(overshoots),
                "mean_overshoot": sum(overshoots) / len(overshoots),
            }
            results.append(row)
            print(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "lora": args.lora,
                   "budget_wording": LENGTH_REWARD_VARIANT, "results": results}, f, indent=2)

    print(f"\n| dataset | budget | accuracy | mean tokens | over % | mean over |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['dataset']} | {r['budget']} | {r['accuracy']:.3f} "
            f"| {r['mean_tokens']:.0f} | {r['overshoot_rate']:.0%} | {r['mean_overshoot']:.0f} |"
        )


if __name__ == "__main__":
    main()

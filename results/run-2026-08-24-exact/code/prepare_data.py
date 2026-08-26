import argparse
import random

from datasets import load_dataset

from rewards import budget_instruction

INSTR = "Let's think step by step and output the final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=8000)
    # The base model writes ~2500 tokens unprompted. Budgets above that ask it to
    # pad, which fights the budgets below it; keeping the range under the natural
    # length makes every example point the same way (shorten) and leaves the
    # 4096 completion cap as real headroom.
    ap.add_argument("--min-budget", type=int, default=200)
    ap.add_argument("--max-budget", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/train")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    # A blank gold answer can never verify, so it is pure reward noise.
    ds = ds.filter(lambda ex: str(ex["answer"]).strip() != "")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))

    def to_example(ex):
        budget = rng.randint(args.min_budget, args.max_budget)
        content = f"{ex['problem']}\n\n{INSTR} {budget_instruction(budget)}"
        return {
            "prompt": [{"role": "user", "content": content}],
            "answer": ex["answer"],
            "budget": budget,
        }

    ds = ds.map(to_example, remove_columns=ds.column_names)
    ds.save_to_disk(args.out)
    print(f"saved {len(ds)} examples to {args.out}")
    print(ds[0]["prompt"][0]["content"][:300])


if __name__ == "__main__":
    main()

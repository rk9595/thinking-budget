import argparse
import random

from datasets import load_dataset

from rewards import budget_instruction
from experiment import build_prompt, read_records
from rewards import LENGTH_REWARD_VARIANT

INSTR = "Let's think step by step and output the final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=8000)
    # Historical defaults. Choose the range explicitly for a new experiment.
    ap.add_argument("--min-budget", type=int, default=200)
    ap.add_argument("--max-budget", type=int, default=2000)
    ap.add_argument("--problem-file", help="frozen training-only JSONL instead of sampling the source dataset")
    ap.add_argument("--budgets", nargs="+", type=int,
                    help="expand every problem at each listed budget, instead of one random budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/train")
    args = ap.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        ap.error("output already exists; choose a new dataset path")
    if args.min_budget <= 0 or args.max_budget < args.min_budget:
        ap.error("invalid budget range")
    if args.budgets and (min(args.budgets) <= 0 or len(set(args.budgets)) != len(args.budgets)):
        ap.error("budgets must be positive and unique")

    if args.problem_file:
        from datasets import Dataset
        problems = read_records(args.problem_file)
        if not problems or len({p['problem_id'] for p in problems}) != len(problems):
            ap.error("training problem file must contain unique problems")
        rng = random.Random(args.seed)
        rows = []
        for ex in problems:
            for budget in args.budgets or [rng.randint(args.min_budget, args.max_budget)]:
                rows.append({"prompt": [{"role": "user", "content": build_prompt(
                    ex['problem'], budget, LENGTH_REWARD_VARIANT)}],
                    "answer": ex['answer'], "budget": budget, "problem_id": ex['problem_id']})
        Dataset.from_list(rows).shuffle(seed=args.seed).save_to_disk(args.out)
        print(f"saved {len(rows)} examples from {len(problems)} frozen training problems")
        return
    if args.budgets:
        ap.error("--budgets requires --problem-file to keep split provenance explicit")

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

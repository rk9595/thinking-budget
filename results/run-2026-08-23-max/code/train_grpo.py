import argparse
import os

from datasets import load_from_disk

import rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--output-dir", default="checkpoints/lcpo-max")
    ap.add_argument("--max-steps", type=int, default=1000)
    # LoRA needs ~10x the full-finetune LR; at 5e-6 the run of 2026-08-22 moved
    # the policy by kl=6e-4 over 1000 steps, i.e. not at all.
    ap.add_argument("--lr", type=float, default=2e-5)
    # "max" matches the "Think for maximum N tokens" prompt in prepare_data.py;
    # "exact" contradicts it and was what the flat 2026-08-22 run trained on.
    ap.add_argument("--length-reward", choices=["exact", "max"], default="max")
    ap.add_argument("--save-steps", type=int, default=25)
    # micro-batch x accum is held at 32 so the effective rollout batch (and thus
    # 4 unique prompts x 8 generations) is unchanged when tuning for memory.
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--vllm-mem", type=float, default=0.20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run on Qwen3-0.6B without vllm to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.model = "Qwen/Qwen3-0.6B"
        args.max_steps = 2
    os.environ["TB_MODEL_ID"] = args.model

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_from_disk(args.data)
    if args.smoke:
        ds = ds.select(range(16))

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_steps=0 if args.smoke else 20,
        per_device_train_batch_size=2 if args.smoke else args.micro_batch,
        gradient_accumulation_steps=1 if args.smoke else 32 // args.micro_batch,
        vllm_gpu_memory_utilization=args.vllm_mem,
        num_generations=2 if args.smoke else 8,
        max_completion_length=64 if args.smoke else 4096,
        beta=0.001,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        bf16=not args.smoke,
        # MPS hits NaN in the trainer's generation path; smoke runs on CPU
        use_cpu=args.smoke,
        use_vllm=not args.smoke,
        vllm_mode="colocate",
        report_to="none" if args.smoke else "wandb",
        run_name=os.path.basename(args.output_dir),
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    length_fn = rewards.length_reward if args.length_reward == "exact" else rewards.length_reward_max
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[rewards.correctness_reward, length_fn],
        args=config,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()

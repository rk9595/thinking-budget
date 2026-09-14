import argparse
import json
import os
from pathlib import Path

import rewards
from experiment import digest, environment, resolve_model, write_json


def validate_data(ds, variant):
    if not len(ds):
        raise ValueError("empty training dataset")
    for i, row in enumerate(ds):
        expected = f"Think for {rewards.WORDING[variant]} {row['budget']} tokens."
        if row['budget'] <= 0 or not row['prompt'][0]['content'].endswith(expected):
            raise ValueError(f"row {i}: budget metadata and prompt/reward wording disagree")


def validate_batch(micro_batch, batch_size, generations, world_size):
    if min(micro_batch, batch_size, generations, world_size) <= 0 or generations < 2:
        raise ValueError("positive batch sizes and at least two generations are required")
    if batch_size % (micro_batch * world_size) or batch_size % generations:
        raise ValueError("batch-size must divide evenly by micro-batch * world-size and generations")
    return batch_size // (micro_batch * world_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--revision")
    ap.add_argument("--output-dir", default="checkpoints/lcpo-exact")
    ap.add_argument("--max-steps", type=int, default=1000)
    # Retained from run 3; the right adapter LR is an empirical setting, not a fixed multiplier.
    ap.add_argument("--lr", type=float, default=2e-5)
    # Run 2 (max) hit the degenerate solution: nothing penalizes finishing early,
    # so the model went uniformly short instead of conditioning on N. Exact
    # penalizes undershoot too, which is what makes the budget a dial.
    ap.add_argument("--length-reward", choices=["exact", "max"], default="exact")
    ap.add_argument("--save-steps", type=int, default=25)
    # --batch-size counts responses, so 32 / 8 generations means four prompt groups.
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32, help="global number of responses per update")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=3e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=.6)
    ap.add_argument("--top-p", type=float, default=.95)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--loss-type", choices=["dapo", "dr_grpo", "grpo"], default="dapo")
    ap.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="group")
    ap.add_argument("--stop-after-step", type=int, help="save and pause at this absolute step; retain full LR schedule")
    ap.add_argument("--init-adapter", help="warm-start a new GRPO run from an SFT adapter")
    ap.add_argument("--vllm-mem", type=float, default=0.20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run on Qwen3-0.6B without vllm to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        if args.model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            args.model = "Qwen/Qwen3-0.6B"
        args.max_steps = 2
        args.batch_size = 2
        args.micro_batch = 2
        args.num_generations = 2
        args.max_completion_length = 64
    os.environ["TB_MODEL_ID"] = args.model
    os.environ["TB_LENGTH_REWARD"] = args.length_reward
    rewards.ALPHA = args.alpha
    if args.alpha <= 0 or args.lora_rank <= 0 or args.max_steps <= 0:
        ap.error("alpha, lora-rank, and max-steps must be positive")
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= args.max_steps:
        ap.error("stop-after-step must be between 1 and max-steps")
    accum = validate_batch(args.micro_batch, args.batch_size, args.num_generations,
                           int(os.environ.get("WORLD_SIZE", "1")))

    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    ds = load_from_disk(args.data)

    # Run 1 trained the Exact reward against a "maximum N" prompt and learned
    # nothing. prepare_data.py bakes the wording into every example, so read it
    # back off the data rather than discovering the mismatch from a flat eval
    # eight hours later.
    # match the instruction phrase, not the bare word - plenty of DeepScaleR
    # problems ask for the "maximum value of" something
    validate_data(ds, args.length_reward)
    if not args.smoke and max(ds['budget']) >= args.max_completion_length:
        ap.error("completion cap must exceed the largest training budget")

    if args.smoke:
        ds = ds.select(range(min(16, len(ds))))

    output = Path(args.output_dir)
    manifest_path = output / "run_manifest.json"
    previous = json.loads(manifest_path.read_text()) if args.resume else None
    requested_revision = args.revision or (previous["spec"]["model_revision"] if previous else None)
    model_id, revision = resolve_model(args.model, requested_revision)
    spec = {k: v for k, v in vars(args).items() if k not in ["resume", "stop_after_step"]}
    spec.update(model_revision=revision, dataset_sha256=digest(ds.to_list()))
    if args.init_adapter:
        import hashlib
        adapter_config = json.loads((Path(args.init_adapter)/"adapter_config.json").read_text())
        if adapter_config["base_model_name_or_path"] != args.model:
            ap.error("initial adapter base model does not match --model")
        with (Path(args.init_adapter)/"adapter_model.safetensors").open("rb") as f:
            spec["initial_adapter_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    if args.resume:
        if previous["spec"] != spec:
            ap.error("resume configuration/data differs from the saved run")
        if args.stop_after_step is not None:
            saved = json.loads((output/"training_status.json").read_text())
            if saved.get("global_step", 0) >= args.stop_after_step:
                ap.error("stop-after-step must exceed the saved checkpoint step")
    elif output.exists() and any(output.iterdir()):
        ap.error("output directory is nonempty; use --resume or a new output directory")

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        warmup_steps=0 if args.smoke else 20,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=accum,
        vllm_gpu_memory_utilization=args.vllm_mem,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        data_seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        model_init_kwargs={"revision": revision},
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
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    length_fn = rewards.length_reward if args.length_reward == "exact" else rewards.length_reward_max
    class PauseAtStep(TrainerCallback):
        def on_step_end(self, config, state, control, **kwargs):
            if args.stop_after_step is not None and state.global_step >= args.stop_after_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    if not args.resume:
        write_json(manifest_path, {"spec": spec, "trainer_config": config.to_dict(),
                   "environment": environment(), "dataset_rows": len(ds),
                   "prompt_groups_per_update": args.batch_size // args.num_generations,
                   "estimated_rollout_responses": args.batch_size * args.max_steps})
    status_path = output / "training_status.json"
    write_json(status_path, {"state": "running"})
    try:
        model = model_id
        if args.init_adapter:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype="auto"), args.init_adapter, is_trainable=True)
            peft_config = None
            config.model_init_kwargs = None
        trainer = GRPOTrainer(model=model, reward_funcs=[rewards.correctness_reward, length_fn],
                              args=config, train_dataset=ds, peft_config=peft_config,
                              callbacks=[PauseAtStep()])
        # Use the exact trainer tokenizer/template when detecting a reasoning prefill.
        rewards._tokenizer = trainer.processing_class
        trainer.train(resume_from_checkpoint=args.resume)
        step = trainer.state.global_step
        checkpoint = (output / f"checkpoint-{step}").resolve()
        if not checkpoint.is_dir():
            raise RuntimeError("final step has no resumable checkpoint; use a save-step boundary")
        write_json(status_path, {"state": "completed" if step >= args.max_steps else "paused",
                                  "global_step": step, "checkpoint": str(checkpoint)})
    except BaseException as exc:
        write_json(status_path, {"state": "failed", "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

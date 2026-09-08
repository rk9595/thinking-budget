"""Supervised budget-conditioning pilot; run only after the reference evaluation passes."""
import argparse
import json
from pathlib import Path

from experiment import digest, environment, read_records, resolve_model, write_json


def render_examples(records, tokenizer):
    """Completed-message chat templates strip CoT; render the user prefix only."""
    examples = []
    for row in records:
        prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        completion = row["completion"][0]["content"]
        # DeepSeek starts <think> in its prompt; L1 may generate that token itself.
        if prompt.rstrip().endswith("<think>") and completion.lstrip().startswith("<think>"):
            completion = completion.lstrip()[len("<think>"):]
        if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        examples.append({"prompt": prompt, "completion": completion})
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/pilot/sft.jsonl")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--revision")
    ap.add_argument("--control-report", required=True)
    ap.add_argument("--out-dir", default="checkpoints/sft-pilot")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report = json.loads(Path(args.control_report).read_text())
    if not report["reference_gate"]["reference_control_detected"]:
        ap.error("reference control was not detected; resolve evaluation before training")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        ap.error("output directory must be new")
    records = read_records(args.data)
    if not records:
        ap.error("empty SFT dataset")
    source = json.loads(Path(args.data).with_suffix(".manifest.json").read_text())
    if source["sft_sha256"] != digest(records):
        ap.error("SFT data differs from verified teacher dataset")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        ap.error("pilot training requires a CUDA GPU")
    model, revision = resolve_model(args.model, args.revision)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    examples = render_examples(records, tok)
    # Fail instead of silently truncating the long-budget demonstrations.
    for i, row in enumerate(examples):
        ids = tok(row["prompt"]+row["completion"], add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_length:
            ap.error(f"example {i} exceeds max-length; inspect it before changing the cap")
    cfg = SFTConfig(output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
                    per_device_train_batch_size=1, gradient_accumulation_steps=16,
                    max_length=args.max_length, packing=False, completion_only_loss=True,
                    bf16=True, gradient_checkpointing=True, seed=args.seed, data_seed=args.seed,
                    logging_steps=1, save_steps=25, save_total_limit=2, report_to="none",
                    model_init_kwargs={"revision": revision, "dtype": "bfloat16"})
    write_json(out/"run_manifest.json", {"config": cfg.to_dict(), "args": vars(args),
               "model_revision": revision, "dataset_sha256": digest(records),
               "teacher_data": source, "control_report": report, "environment": environment()})
    write_json(out/"training_status.json", {"state": "running"})
    try:
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(examples),
                             processing_class=tok, peft_config=LoraConfig(
                                 r=args.rank, lora_alpha=2*args.rank, target_modules="all-linear",
                                 task_type="CAUSAL_LM"))
        trainer.train()
        trainer.save_model(str(out/"final"))
        tok.save_pretrained(out/"final")
        write_json(out/"training_status.json", {"state": "completed",
                   "global_step": trainer.state.global_step, "adapter": str((out/"final").resolve())})
    except BaseException as exc:
        write_json(out/"training_status.json", {"state": "failed", "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()

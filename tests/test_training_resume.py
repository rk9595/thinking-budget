"""Exercise real TRL checkpoint/pause/resume with a tiny random model on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_grpo_pause_resume_and_configuration_guard(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from experiment import build_prompt

    model_dir, data_dir, output = tmp_path/"model", tmp_path/"data", tmp_path/"run"
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3,
             "assistant": 4, "42": 5, "Think": 6, "tokens": 7}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", bos_token="<bos>",
                                 eos_token="<eos>", pad_token="<pad>")
    tok.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tok.save_pretrained(model_dir)
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=2, bos_token_id=1, eos_token_id=2, pad_token_id=3))
    model.save_pretrained(model_dir)
    Dataset.from_list([{"prompt": [{"role": "user", "content": build_prompt(f"q{i}", 100)}],
                        "answer": "42", "budget": 100} for i in range(4)]).save_to_disk(str(data_dir))
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root/"train_grpo.py"), "--smoke", "--model", str(model_dir),
           "--data", str(data_dir), "--output-dir", str(output), "--save-steps", "1"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    def run(extra):
        return subprocess.run(cmd+extra, env=env, capture_output=True, text=True, timeout=120)
    first = run(["--stop-after-step", "1"])
    assert first.returncode == 0, first.stdout + first.stderr
    status = json.loads((output/"training_status.json").read_text())
    assert status["state"] == "paused" and status["global_step"] == 1
    assert (Path(status["checkpoint"])/"optimizer.pt").exists()
    mismatch = run(["--resume", "--lr", "0.001"])
    assert mismatch.returncode != 0 and "differs" in mismatch.stderr
    resumed = run(["--resume"])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    status = json.loads((output/"training_status.json").read_text())
    assert status["state"] == "completed" and status["global_step"] == 2

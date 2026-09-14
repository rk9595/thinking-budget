import json

import pytest

from experiment import write_json
from regrade_eval import regrade


def test_regrade_preserves_original_and_rejects_unfinished_prefill(tmp_path):
    row = {"dataset": "dev", "problem_id": "a", "budget": 512, "seed": 42,
           "text": "Maybe \\boxed{42}", "answer": "42", "rendered_prompt": "assistant<think>\n",
           "total_tokens": 32, "reasoning_tokens": None, "finish_reason": "length", "correct": True}
    raw = tmp_path / "original.samples.jsonl"
    raw.write_text(json.dumps(row) + "\n")
    write_json(tmp_path / "original.manifest.json", {
        "problems": [{"dataset": "dev", "problem_id": "a"}],
        "config": {"budgets": [512], "seeds": [42]}})
    source = tmp_path / "original.json"
    write_json(source, {"manifest": "original.manifest.json", "samples": raw.name})
    before = raw.read_bytes()
    output = tmp_path / "audit" / "corrected.json"
    regrade(source, output)
    corrected = json.loads(output.with_suffix(".samples.jsonl").read_text())
    assert corrected["legacy_correct"] is True and corrected["correct"] is False
    assert raw.read_bytes() == before
    with pytest.raises(ValueError, match="never overwritten"):
        regrade(source, output)

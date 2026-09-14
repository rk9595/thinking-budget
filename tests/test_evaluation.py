import json
import sys
from types import SimpleNamespace

import pytest

import eval_budget
from compare_results import paired_difference, reference_gate, validate_samples
from experiment import build_prompt
from train_grpo import validate_batch, validate_data


def sample(problem, budget, length, seed=42, finish="stop"):
    return {"dataset": "math500", "problem_id": problem, "budget": budget, "seed": seed,
            "total_tokens": length, "reasoning_tokens": None, "correct": True,
            "finish_reason": finish}


def test_spread_cannot_mistake_reversed_or_flat_budget_response_for_control():
    base = [sample(str(i), b, 700) for i in range(4) for b in [512, 1024, 3600]]
    exact = [sample(str(i), b, b) for i in range(4) for b in [512, 1024, 3600]]
    reverse = [sample(str(i), b, 4112-b) for i in range(4) for b in [512, 1024, 3600]]
    assert reference_gate(exact, base)["reference_control_detected"]
    assert not reference_gate(base, base)["reference_control_detected"]
    assert not reference_gate(reverse, base)["reference_control_detected"]


def test_truncated_outputs_cannot_pass_reference_gate():
    base = [sample("a", b, 700) for b in [512, 1024, 3600]]
    clipped = [sample("a", b, b, finish="length") for b in [512, 1024, 3600]]
    assert not reference_gate(clipped, base)["reference_control_detected"]


def test_pairing_rejects_missing_seeds_and_duplicate_samples():
    a = sample("a", 512, 600)
    with pytest.raises(ValueError, match="identical"):
        paired_difference([a], [sample("a", 512, 600, seed=43)])
    with pytest.raises(ValueError, match="duplicate"):
        paired_difference([a, a], [a])


def test_matching_partial_results_do_not_count_as_complete_evaluation():
    manifest = {"problems": [{"dataset": "math500", "problem_id": "a"}],
                "config": {"budgets": [512, 1024], "seeds": [42]}}
    with pytest.raises(ValueError, match="complete"):
        validate_samples([sample("a", 512, 500)], manifest)


def test_bootstrap_resamples_problems_not_seeds():
    candidate = [sample("a", 512, 700, seed=s) for s in [1, 2, 3]]
    base = [sample("a", 512, 600, seed=s) for s in [1, 2, 3]]
    result = paired_difference(candidate, base)
    lengths = next(r for r in result if r["metric"] == "total_tokens")
    assert lengths["n_problems"] == 1
    assert lengths["bootstrap_95pct_ci"] == [100, 100]


def test_missing_think_delimiter_is_unknown():
    tok = SimpleNamespace(encode=lambda text, **kw: {"<think>": [1], "</think>": [2, 3]}[text])
    assert eval_budget.reasoning_counts([1, 8, 9, 2, 3, 7], tok) == (2, 1)
    assert eval_budget.reasoning_counts([8, 9, 2, 3, 7], tok) == (2, 1)
    assert eval_budget.reasoning_counts([8, 9], tok) == (None, None)


def test_training_checks_every_row_and_matching_budget():
    ds = [{"budget": 512, "prompt": [{"content": build_prompt("q", 512)}]},
          {"budget": 1024, "prompt": [{"content": build_prompt("q", 512)}]}]
    with pytest.raises(ValueError, match="row 1"):
        validate_data(ds, "exact")
    assert validate_batch(2, 32, 8, 1) == 16
    with pytest.raises(ValueError):
        validate_batch(3, 32, 8, 1)


def test_eval_writes_replayable_samples_and_refuses_overwrite(tmp_path, monkeypatch):
    class Tokenizer:
        chat_template = "test"

        def encode(self, text, **kwargs):
            return [1] if text == "<think>" else [2]

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    captured = []

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params, **kwargs):
            captured.extend(params)
            result = SimpleNamespace(text="reasoning</think>\\boxed{42}", token_ids=[7, 2, 8],
                                     finish_reason="stop", stop_reason=None)
            return [SimpleNamespace(outputs=[result]) for p in prompts]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(eval_budget, "resolve_model", lambda *a: ("fake", "frozen-sha"))
    monkeypatch.setattr(eval_budget, "select_problems", lambda *a: [
        {"dataset": "math500", "problem_id": "abcdef12", "problem": "6*7?", "answer": "42"}])
    output = tmp_path/"eval.json"
    monkeypatch.setattr(sys, "argv", ["eval_budget.py", "--out", str(output),
                                     "--budgets", "512", "1024", "--seeds", "42", "43"])
    eval_budget.main()
    summary = json.loads(output.read_text())
    rows = [json.loads(s) for s in output.with_suffix(".samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(r["correct"] for r in rows)
    assert rows[0]["reasoning_tokens"] == 1
    assert rows[0]["total_tokens"] == 3
    assert captured[0]["seed"] == captured[1]["seed"]
    assert captured[0]["seed"] != captured[2]["seed"]
    assert summary["results"][0]["n"] == 2
    assert summary["results"][0]["n_problems"] == 1
    with pytest.raises(SystemExit):
        eval_budget.main()

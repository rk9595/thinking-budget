import pytest

from prepare_pilot import filter_teacher, split_problems
from train_sft import render_examples


def test_split_deduplicates_before_assigning_train_and_dev():
    rows = [{"problem": " SAME  QUESTION ", "answer": "1"},
            {"problem": "same question", "answer": "1"},
            {"problem": "benchmark", "answer": "2"},
            {"problem": "different", "answer": "3"}]
    train, dev = split_problems(rows, [" BENCHMARK "], 1, 1, 42)
    assert train[0]["problem_id"] != dev[0]["problem_id"]
    assert len(train+dev) == 2
    assert not any(r["problem"] == "benchmark" for r in train+dev)


def test_teacher_filter_requires_correct_complete_budget_sets():
    problems = [{"problem_id": "a", "problem": "q", "answer": "42"}]
    good = [{**problems[0], "budget": b, "total_tokens": b, "reasoning_tokens": b-10,
             "finish_reason": "stop", "text": "<think>work</think>\\boxed{42}"} for b in [512, 1024]]
    assert len(filter_teacher(good, problems, [512, 1024], .25)) == 2
    assert not filter_teacher(good[:1], problems, [512, 1024], .25)
    bad = [good[0], {**good[1], "finish_reason": "length"}]
    assert not filter_teacher(bad, problems, [512, 1024], .25)
    with pytest.raises(ValueError, match="outside"):
        filter_teacher([{**good[0], "problem_id": "dev"}], problems, [512, 1024], .25)


def test_sft_keeps_reasoning_that_full_chat_template_would_strip():
    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1, "never render the completed assistant through the template"
            return "user question assistant<think>\n"

    row = {"prompt": [{"role": "user", "content": "q"}],
           "completion": [{"role": "assistant", "content": "<think>preserve reasoning</think>42"}]}
    example = render_examples([row], Tokenizer())[0]
    assert example["completion"] == "preserve reasoning</think>42<eos>"
    assert (example["prompt"]+example["completion"]).count("<think>") == 1

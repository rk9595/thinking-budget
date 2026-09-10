import os

ALPHA = 3e-4

# The prompt wording has to match the reward variant, and training, eval and the
# inference profile run as separate processes on the box. Run 1 shipped
# "maximum" against the Exact reward and produced a flat, useless model, so the
# wording lives in one place and train_grpo.py refuses to start on a mismatch.
WORDING = {"exact": "exactly", "max": "maximum"}
LENGTH_REWARD_VARIANT = os.environ.get("TB_LENGTH_REWARD", "exact")


def budget_instruction(budget):
    return f"Think for {WORDING[LENGTH_REWARD_VARIANT]} {budget} tokens."

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(os.environ.get(
            "TB_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))
    return _tokenizer


def _text(completion):
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


def is_correct(completion_text, gold_answer, *, prefilled_think=False):
    from math_verify import parse, verify

    try:
        if prefilled_think and "</think>" not in completion_text:
            return False
        gold = parse(f"${gold_answer}$")
        # Do not grade an intermediate answer from inside a completed reasoning trace.
        final = completion_text.split("</think>", 1)[-1]
        if "<think>" in final:
            return False
        pred = parse(final)
        return bool(verify(gold, pred))
    except Exception:
        return False


def correctness_reward(completions, answer, prompts=None, **kwargs):
    prefixes = [""] * len(completions)
    if prompts is not None:
        prefixes = [p if isinstance(p, str) else _get_tokenizer().apply_chat_template(
            p, tokenize=False, add_generation_prompt=True) for p in prompts]
    return [1.0 if is_correct(_text(c), a, prefilled_think=p.rstrip().endswith("<think>")) else 0.0
            for c, a, p in zip(completions, answer, prefixes)]


def length_reward(completions, budget, completion_ids=None, **kwargs):
    if completion_ids is not None:
        lengths = [len(ids) for ids in completion_ids]
    else:
        tok = _get_tokenizer()
        lengths = [len(tok(_text(c), add_special_tokens=False)["input_ids"]) for c in completions]
    return [-ALPHA * abs(b - n) for b, n in zip(budget, lengths)]


def length_reward_max(completions, budget, completion_ids=None, **kwargs):
    if completion_ids is not None:
        lengths = [len(ids) for ids in completion_ids]
    else:
        tok = _get_tokenizer()
        lengths = [len(tok(_text(c), add_special_tokens=False)["input_ids"]) for c in completions]
    return [-ALPHA * max(0, n - b) for b, n in zip(budget, lengths)]

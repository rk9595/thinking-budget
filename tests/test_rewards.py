import math

from rewards import ALPHA, correctness_reward, length_reward, length_reward_max

CHAT = lambda text: [{"role": "assistant", "content": text}]


def test_correct_boxed_answer():
    comps = [CHAT("The sum is 6·7=42, so the answer is \\boxed{42}.")]
    assert correctness_reward(comps, answer=["42"]) == [1.0]


def test_correct_equivalent_form():
    comps = [CHAT("\\boxed{\\frac{1}{2}}")]
    assert correctness_reward(comps, answer=["0.5"]) == [1.0]


def test_incorrect_answer():
    comps = [CHAT("\\boxed{41}")]
    assert correctness_reward(comps, answer=["42"]) == [0.0]


def test_no_answer_at_all():
    comps = [CHAT("I ran out of tokens before finishing")]
    assert correctness_reward(comps, answer=["42"]) == [0.0]


def test_plain_string_completion():
    assert correctness_reward(["\\boxed{7}"], answer=["7"]) == [1.0]


def test_batch_mixed():
    comps = [CHAT("\\boxed{1}"), CHAT("\\boxed{2}")]
    assert correctness_reward(comps, answer=["1", "3"]) == [1.0, 0.0]


def test_length_exact_on_budget():
    ids = [list(range(100))]
    r = length_reward([CHAT("x")], budget=[100], completion_ids=ids)
    assert r == [0.0]


def test_length_exact_over_budget():
    ids = [list(range(300))]
    r = length_reward([CHAT("x")], budget=[100], completion_ids=ids)
    assert math.isclose(r[0], -ALPHA * 200)


def test_length_exact_under_budget_also_penalized():
    ids = [list(range(50))]
    r = length_reward([CHAT("x")], budget=[100], completion_ids=ids)
    assert math.isclose(r[0], -ALPHA * 50)


def test_length_max_under_budget_free():
    ids = [list(range(50))]
    r = length_reward_max([CHAT("x")], budget=[100], completion_ids=ids)
    assert r == [0.0]


def test_length_max_over_budget():
    ids = [list(range(150))]
    r = length_reward_max([CHAT("x")], budget=[100], completion_ids=ids)
    assert math.isclose(r[0], -ALPHA * 50)

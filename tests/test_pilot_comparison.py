from compare_pilot import pilot_gate


def rows(controlled=True, correct=True):
    return [{"dataset": "dev", "problem_id": str(i), "budget": b, "seed": 42,
             "total_tokens": b if controlled else 2000, "correct": correct,
             "finish_reason": "stop", "reasoning_tokens": 100}
            for i in range(4) for b in [512, 1024, 3600]]


def test_pilot_needs_both_control_and_accuracy():
    base = rows(controlled=False)
    assert pilot_gate(rows(), base)["ready_for_grpo_pilot"]
    assert not pilot_gate(rows(controlled=False), base)["ready_for_grpo_pilot"]
    inaccurate = rows()
    inaccurate[-1]["correct"] = False  # 25-point high-budget accuracy drop.
    gate = pilot_gate(inaccurate, base)
    assert not gate["ready_for_grpo_pilot"]
    assert not gate["checks"]["high_budget_accuracy_loss_at_most_10_points"]

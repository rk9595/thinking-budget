import json

from experiment import write_json
from paid_sanity import RUN, artifact_hashes, backup_verified
from sanity_sft import select_ids
from audit_sanity import summarize_errors


def test_selection_is_deterministic_and_disjoint():
    rows = [{"problem_id": f"p{i}"} for i in range(50)] * 3
    train, probe = select_ids(rows)
    assert len(train) == 24 and len(probe) == 12
    assert not set(train) & set(probe)
    assert (train, probe) == select_ids(list(reversed(rows)))


def test_artifact_verification_rejects_missing_and_changed_data(tmp_path):
    directory = tmp_path / "results" / RUN
    write_json(directory / "job_status.json", {"exit_code": 0})
    assert not backup_verified(tmp_path)
    write_json(directory / "artifact_hashes.json", artifact_hashes(tmp_path))
    assert backup_verified(tmp_path)
    write_json(directory / "job_status.json", {"exit_code": 1})
    assert not backup_verified(tmp_path)


def test_frozen_plan_matches_local_data():
    from pathlib import Path
    from experiment import digest, read_records
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "results" / RUN / "plan.json").read_text())
    assert plan["problem_sha256"] == digest(read_records(root / "results" / RUN / "problems.jsonl"))
    assert not set(plan["train_ids"]) & set(plan["probe_ids"])
    assert plan["stage_one_spending_limit_usd"] == 3


def test_error_audit_keeps_outliers_and_handles_perfect_lengths():
    rows = [{"dataset": "train", "budget": 100, "problem_id": str(i), "problem": "q",
             "total_tokens": n, "text": "</think> answer", "correct": True,
             "finish_reason": "stop"} for i, n in enumerate([100, 100, 400])]
    result = summarize_errors(rows)[0]
    assert result["mean_absolute_relative_error"] == 1
    assert result["median_absolute_relative_error"] == 0
    assert result["largest_two_share_of_absolute_error"] == 1
    assert result["outliers"][0]["total_tokens"] == 400
    assert summarize_errors(rows[:2])[0]["largest_two_share_of_absolute_error"] == 0


def test_saved_sanity_results_do_not_unlock_stage_two():
    from pathlib import Path
    from compare_pilot import load_evaluation
    from sanity_sft import train_gate
    root = Path(__file__).resolve().parents[1] / "results" / RUN
    base = load_evaluation(root / "base.json")[2]
    base = [r for r in base if r["dataset"] == "sanity_train"]
    for name in ["token", "sequence"]:
        rows = load_evaluation(root / (name + ".json"))[2]
        gate = train_gate([r for r in rows if r["dataset"] == "sanity_train"], base)
        assert not gate["passed"]
        assert [k for k, passed in gate["checks"].items() if not passed] == ["length_error_under_35pct_each_budget"]

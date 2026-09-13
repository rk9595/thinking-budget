import json

from experiment import write_json
from paid_sanity import RUN, artifact_hashes, backup_verified
from sanity_sft import select_ids


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

from decode_sanity import decision


def test_guard_run_name_is_scoped_and_rejects_paths():
    import os
    import subprocess
    import sys
    for name, allowed in [("decode-2026-09-14", True), ("../other", False), ("/", False)]:
        result = subprocess.run([sys.executable, "-c", "from paid_sanity import RUN; print(RUN)"],
            capture_output=True, text=True, env={**os.environ, "TB_DIAGNOSTIC_RUN": name})
        assert (result.returncode == 0) == allowed
        if allowed:
            assert result.stdout.strip() == name


def test_replication_requires_both_fresh_seeds_and_does_not_unlock_training():
    result = decision({"token": [True, False], "sequence": [True, True]},
                      {"token": True, "sequence": False})
    assert result["replicated_training_gate_arms"] == ["sequence"]
    assert not result["stage_two_permitted"]


def test_greedy_is_secondary_not_a_replacement_for_failed_gate():
    result = decision({"token": [False, False], "sequence": [False, False]},
                      {"token": True, "sequence": False})
    assert "decoding sensitivity" in result["recommendation"]
    assert not result["stage_two_permitted"]


def test_no_signal_recommends_stopping_instead_of_more_training():
    result = decision({"token": [False, False], "sequence": [False, False]},
                      {"token": False, "sequence": False})
    assert result["recommendation"].startswith("pause")
    assert not result["stage_two_permitted"]

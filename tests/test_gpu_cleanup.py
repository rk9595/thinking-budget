from types import SimpleNamespace

import pytest

from cleanup_gpu import destroy_and_verify


def test_destroy_uses_confirmation_flag_and_verifies_only_selected_target():
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["check"]
        return SimpleNamespace(stdout='[{"id": 99}]')

    destroy_and_verify(42, run=run)
    assert calls[0][-5:] == ["destroy", "instance", "42", "--yes", "--raw"]
    assert calls[1][-3:] == ["show", "instances", "--raw"]


def test_successful_command_exit_is_not_proof_of_deletion():
    def run(args, **kwargs):
        return SimpleNamespace(stdout='[{"id": 42, "actual_status": "exited"}]')

    with pytest.raises(RuntimeError, match="still exists"):
        destroy_and_verify(42, run=run)


def test_cleanup_rejects_ambiguous_target():
    with pytest.raises(ValueError):
        destroy_and_verify(0)

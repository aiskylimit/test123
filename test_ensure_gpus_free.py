"""Tests for ensure_gpus_free and wait_for_step process management."""

import os
import signal
import subprocess
import time
from unittest import mock

import pytest

from run_smoke_tests_v3 import ensure_gpus_free, wait_for_step


class FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestNvidiaSmiParsing:
    """Test that nvidia-smi output is parsed correctly in all edge cases."""

    def test_no_gpu_processes(self):
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep"):
            mock_run.return_value = FakeCompletedProcess(stdout="")
            assert ensure_gpus_free(max_attempts=1) is True

    def test_no_gpu_processes_trailing_newline(self):
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep"):
            mock_run.return_value = FakeCompletedProcess(stdout="\n")
            assert ensure_gpus_free(max_attempts=1) is True

    def test_no_gpu_processes_whitespace_only(self):
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep"):
            mock_run.return_value = FakeCompletedProcess(stdout="  \n  \n  ")
            assert ensure_gpus_free(max_attempts=1) is True

    def test_single_pid(self):
        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="12345\n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=12345), \
             mock.patch("os.killpg"), \
             mock.patch("os.kill"):
            assert ensure_gpus_free(max_attempts=2) is True

    def test_duplicate_pids_deduplicated(self):
        """One process on 8 GPUs: nvidia-smi reports same PID 8 times."""
        killed_pids = []
        killed_groups = []

        def fake_getpgid(pid):
            return pid

        def fake_killpg(pgid, sig):
            killed_groups.append((pgid, sig))

        def fake_kill(pid, sig):
            killed_pids.append((pid, sig))

        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(
                    stdout="12345\n12345\n12345\n12345\n12345\n12345\n12345\n12345\n"
                )
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", side_effect=fake_getpgid), \
             mock.patch("os.killpg", side_effect=fake_killpg), \
             mock.patch("os.kill", side_effect=fake_kill):
            assert ensure_gpus_free(max_attempts=2) is True

        sigterm_groups = [(pg, s) for pg, s in killed_groups if s == signal.SIGTERM]
        assert len(sigterm_groups) == 1, f"Should SIGTERM group only once, got {len(sigterm_groups)}"

    def test_multiple_distinct_pids(self):
        """Multiple different processes on GPUs."""
        killed_groups = []

        def fake_getpgid(pid):
            return pid

        def fake_killpg(pgid, sig):
            killed_groups.append((pgid, sig))

        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="111\n222\n333\n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", side_effect=fake_getpgid), \
             mock.patch("os.killpg", side_effect=fake_killpg), \
             mock.patch("os.kill"):
            assert ensure_gpus_free(max_attempts=2) is True

        sigterm_groups = [(pg, s) for pg, s in killed_groups if s == signal.SIGTERM]
        assert len(sigterm_groups) == 3

    def test_nvidia_smi_failure(self):
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep"):
            mock_run.return_value = FakeCompletedProcess(
                returncode=1, stderr="NVIDIA-SMI has failed"
            )
            assert ensure_gpus_free(max_attempts=2) is False

    def test_max_attempts_exhausted(self):
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=999), \
             mock.patch("os.killpg"), \
             mock.patch("os.kill"):
            mock_run.return_value = FakeCompletedProcess(stdout="999\n")
            assert ensure_gpus_free(max_attempts=3) is False

    def test_pids_with_extra_whitespace(self):
        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="  12345  \n  67890  \n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=12345), \
             mock.patch("os.killpg"), \
             mock.patch("os.kill"):
            assert ensure_gpus_free(max_attempts=2) is True


class TestKillErrorHandling:
    """Test that kill errors are handled gracefully."""

    def test_process_already_dead(self):
        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="12345\n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", side_effect=ProcessLookupError), \
             mock.patch("os.kill", side_effect=ProcessLookupError):
            assert ensure_gpus_free(max_attempts=2) is True

    def test_permission_error(self):
        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="12345\n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", side_effect=PermissionError("Operation not permitted")), \
             mock.patch("os.kill", side_effect=PermissionError("Operation not permitted")):
            assert ensure_gpus_free(max_attempts=2) is True

    def test_non_integer_pid(self):
        """If nvidia-smi somehow returns non-integer, int() raises ValueError."""
        call_count = [0]
        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return FakeCompletedProcess(stdout="not_a_pid\n")
            return FakeCompletedProcess(stdout="")

        with mock.patch("run_smoke_tests_v3.subprocess.run", side_effect=fake_run), \
             mock.patch("run_smoke_tests_v3.time.sleep"):
            assert ensure_gpus_free(max_attempts=2) is True

    def test_small_sleep_between_no_crash(self):
        """sleep_between < 5 must not crash with negative sleep."""
        with mock.patch("run_smoke_tests_v3.subprocess.run") as mock_run, \
             mock.patch("run_smoke_tests_v3.time.sleep") as mock_sleep:
            mock_run.return_value = FakeCompletedProcess(stdout="")
            ensure_gpus_free(max_attempts=1, sleep_between=2)
            for call_args in mock_sleep.call_args_list:
                assert call_args[0][0] >= 0, f"Negative sleep: {call_args[0][0]}"


class TestWaitForStepKill:
    """Test the kill escalation in wait_for_step."""

    def test_sigterm_works(self, tmp_path):
        csv_path = tmp_path / "smoke_metrics.csv"
        csv_path.write_text("step\n7000\n")

        proc = mock.MagicMock()
        proc.poll.side_effect = [None, None]
        proc.pid = 99999
        proc.wait.return_value = None

        with mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.killpg") as mock_killpg:
            result = wait_for_step(proc, str(csv_path), 6500, poll_interval=0)

        assert result == 7000
        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=60)

    def test_sigterm_timeout_escalates_to_sigkill(self, tmp_path):
        csv_path = tmp_path / "smoke_metrics.csv"
        csv_path.write_text("step\n7000\n")

        proc = mock.MagicMock()
        proc.poll.side_effect = [None, None]
        proc.pid = 99999
        proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 60), None]

        with mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.killpg") as mock_killpg:
            result = wait_for_step(proc, str(csv_path), 6500, poll_interval=0)

        assert result == 7000
        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(99999, signal.SIGTERM)
        mock_killpg.assert_any_call(99999, signal.SIGKILL)

    def test_sigkill_timeout_does_not_crash(self, tmp_path):
        csv_path = tmp_path / "smoke_metrics.csv"
        csv_path.write_text("step\n7000\n")

        proc = mock.MagicMock()
        proc.poll.side_effect = [None, None]
        proc.pid = 99999
        proc.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 60),
            subprocess.TimeoutExpired("cmd", 30),
        ]

        with mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", return_value=99999), \
             mock.patch("os.killpg"):
            result = wait_for_step(proc, str(csv_path), 6500, poll_interval=0)

        assert result == 7000

    def test_process_exits_before_target(self, tmp_path):
        csv_path = tmp_path / "smoke_metrics.csv"
        csv_path.write_text("step\n100\n")

        proc = mock.MagicMock()
        proc.poll.return_value = 0

        result = wait_for_step(proc, str(csv_path), 6500)
        assert result == -1

    def test_process_already_dead_when_killing(self, tmp_path):
        csv_path = tmp_path / "smoke_metrics.csv"
        csv_path.write_text("step\n7000\n")

        proc = mock.MagicMock()
        proc.poll.side_effect = [None, None]
        proc.pid = 99999
        proc.wait.return_value = None

        with mock.patch("run_smoke_tests_v3.time.sleep"), \
             mock.patch("os.getpgid", side_effect=ProcessLookupError), \
             mock.patch("os.killpg", side_effect=ProcessLookupError):
            result = wait_for_step(proc, str(csv_path), 6500, poll_interval=0)

        assert result == 7000

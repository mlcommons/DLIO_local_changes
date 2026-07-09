"""Tests for the per-node page-cache flush that precedes the read_threads
memory guard (mlcommons/storage #741).

The helper under test, ``_flush_page_caches_before_memory_guard``, sits
inside ``ConfigArguments.validate`` and runs immediately before
``psutil.virtual_memory()`` is sampled.  It:

  1. Does nothing when MPI is not initialized (child processes, test
     harness paths).
  2. Otherwise, on ``local_rank() == 0`` only, invokes
     ``sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`` via
     ``subprocess.run`` with a timeout from
     ``_resolve_drop_caches_timeout()``.
  3. Barriers all ranks so non-leaders see post-drop memory.
  4. Fails open: every subprocess/MPI failure is swallowed so the
     guard downstream still runs (matching the fail-open posture of
     the per-epoch flush in ``main.py``).

Rationale for the ``local_rank()``-only gate (not ``MPI.node()``) is
in the helper's docstring — the short version is that ``MPI.node()``
is unreliable under ``--map-by node`` (storage#669 / PR #675) while
``local_rank()`` derives from ``MPI.COMM_TYPE_SHARED`` and is
independent of rank assignment.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from dlio_benchmark.common.enumerations import MPIState
from dlio_benchmark.utils.config import _flush_page_caches_before_memory_guard


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _fake_mpi(
    *, state=MPIState.MPI_INITIALIZED, local_rank=0, barrier_raises=None
):
    """Build a fake ``DLIOMPI`` for injection.

    Only the surface actually consumed by the helper is populated:
    ``mpi_state``, ``local_rank()``, and ``comm().Barrier()``.
    """
    mpi = MagicMock(name="DLIOMPI")
    mpi.mpi_state = state
    mpi.local_rank.return_value = local_rank
    if barrier_raises is not None:
        mpi.comm.return_value.Barrier.side_effect = barrier_raises
    return mpi


# ---------------------------------------------------------------------------
# MPI-state gating
# ---------------------------------------------------------------------------

class TestMpiStateGating:
    """The helper is a no-op unless MPI is fully initialized."""

    @pytest.mark.parametrize(
        "state", [MPIState.UNINITIALIZED, MPIState.CHILD_INITIALIZED]
    )
    def test_noop_when_mpi_not_initialized(self, state):
        """No subprocess, no barrier — otherwise child/harness contexts
        would hit comm() which raises."""
        mpi = _fake_mpi(state=state)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        run.assert_not_called()
        mpi.comm.assert_not_called()


# ---------------------------------------------------------------------------
# local_rank() gate
# ---------------------------------------------------------------------------

class TestLocalRankGate:
    """The flush runs only on local_rank == 0; all ranks barrier."""

    def test_local_rank_zero_runs_drop_caches(self):
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        run.assert_called_once()
        # Ensure argv is exactly the drop_caches command (matches
        # main.py's per-epoch invocation — one shared privilege surface).
        (argv,), kwargs = run.call_args
        assert argv == [
            "sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches",
        ]
        assert kwargs["check"] is False
        assert isinstance(kwargs["timeout"], int) and kwargs["timeout"] >= 1
        assert kwargs["stdin"] is subprocess.DEVNULL
        # All ranks barrier — even the leader — so both leaders and
        # non-leaders read virtual_memory() after the drop.
        mpi.comm.return_value.Barrier.assert_called_once()

    @pytest.mark.parametrize("rank", [1, 2, 7, 15])
    def test_non_leader_ranks_skip_subprocess_but_still_barrier(self, rank):
        """Non-leaders must not spawn sudo (redundant, wasteful) but MUST
        barrier, otherwise the leader's drop could race the non-leader's
        virtual_memory() read."""
        mpi = _fake_mpi(local_rank=rank)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        run.assert_not_called()
        mpi.comm.return_value.Barrier.assert_called_once()


# ---------------------------------------------------------------------------
# Fail-open subprocess semantics
# ---------------------------------------------------------------------------

class TestSubprocessFailureIsSwallowed:
    """Every subprocess failure mode is fail-open so the memory guard
    downstream still runs.  Matches the per-epoch flush posture in
    main.py: a failed drop degrades gracefully, it does not abort the
    benchmark."""

    @pytest.mark.parametrize("exc", [
        subprocess.TimeoutExpired(cmd="sudo", timeout=30),
        FileNotFoundError("sudo not installed"),
        PermissionError("sudo -n refused"),
        OSError("some OS error"),
        RuntimeError("unexpected"),
    ])
    def test_subprocess_exception_swallowed(self, exc):
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run", side_effect=exc):
            # Must NOT raise.
            _flush_page_caches_before_memory_guard(mpi)
        # Barrier still runs — non-leaders elsewhere depend on it.
        mpi.comm.return_value.Barrier.assert_called_once()

    def test_nonzero_exit_code_not_raised(self):
        """``check=False`` means a non-zero exit (sudo refused, kernel
        rejected the write) returns a CompletedProcess rather than
        raising CalledProcessError — the guard downstream then runs
        normally."""
        completed = subprocess.CompletedProcess(
            args=["sudo"], returncode=1, stdout="", stderr="sudo: no tty"
        )
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run", return_value=completed) as run:
            _flush_page_caches_before_memory_guard(mpi)
        run.assert_called_once()
        mpi.comm.return_value.Barrier.assert_called_once()


# ---------------------------------------------------------------------------
# Fail-open MPI barrier
# ---------------------------------------------------------------------------

class TestBarrierFailureIsSwallowed:
    """A rogue Barrier() failure must not mask the memory-guard check —
    the guard is a correctness check; the flush is an optimization."""

    def test_barrier_exception_swallowed_on_leader(self):
        mpi = _fake_mpi(local_rank=0, barrier_raises=RuntimeError("mpi down"))
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)  # must not raise
        run.assert_called_once()

    def test_barrier_exception_swallowed_on_non_leader(self):
        mpi = _fake_mpi(local_rank=3, barrier_raises=RuntimeError("mpi down"))
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)  # must not raise
        run.assert_not_called()


# ---------------------------------------------------------------------------
# Timeout comes from _resolve_drop_caches_timeout (shared source of truth)
# ---------------------------------------------------------------------------

class TestTimeoutPlumbing:
    """The flush timeout is the one operators already tune via
    ``DLIO_DROP_CACHES_TIMEOUT`` for the per-epoch flush.  Introducing a
    second knob would double the operational surface for no gain."""

    def test_timeout_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("DLIO_DROP_CACHES_TIMEOUT", raising=False)
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        _, kwargs = run.call_args
        # Default is 30s (see test_drop_caches_timeout.py::
        # test_default_constant_matches_storage_391_history).
        assert kwargs["timeout"] == 30

    def test_timeout_honors_env_override(self, monkeypatch):
        monkeypatch.setenv("DLIO_DROP_CACHES_TIMEOUT", "300")
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        _, kwargs = run.call_args
        assert kwargs["timeout"] == 300

    def test_timeout_env_bad_value_falls_back_to_default(self, monkeypatch):
        """Unparseable env value must not crash; matches the tolerant
        posture of ``_resolve_drop_caches_timeout``."""
        monkeypatch.setenv("DLIO_DROP_CACHES_TIMEOUT", "not-a-number")
        mpi = _fake_mpi(local_rank=0)
        with patch("subprocess.run") as run:
            _flush_page_caches_before_memory_guard(mpi)
        _, kwargs = run.call_args
        assert kwargs["timeout"] == 30

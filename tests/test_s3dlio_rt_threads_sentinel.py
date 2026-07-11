"""
Regression tests for the DLIO -> s3dlio S3DLIO_RT_THREADS miscomputation
(root-caused 2026-07-11, see s3dlio's
docs/investigation/DLIO_UNET3D_DATAGEN_BOTTLENECK_INVESTIGATION_2026-07-10.md).

Before this fix, ObjStoreLibStorage.__init__ derived
`S3DLIO_RT_THREADS = write_threads * 3 // 2` (cap 128) as a hint to
s3dlio's global Tokio runtime.  The derivation ran at
[main.py:132](../dlio_benchmark/main.py#L132) — BEFORE line 505's
`derive_configurations()` auto-sizes `write_threads` from its dataclass
default sentinel (1) to the real workload-appropriate value (e.g. 32
for a 28-core NP=1 S3 workload).  Result: `S3DLIO_RT_THREADS` was set
to `1 * 3 // 2 = 1`, s3dlio built its Tokio runtime with ONE worker,
and every concurrent multipart-upload part serialized on it — measured
~214 MB/s at NP=1 vs. ~1928 MB/s once the runtime is correctly sized.

Fix: `ObjStoreLibStorage._configure_s3dlio_runtime_env()` now skips
the auto-derive when `write_threads` is still at the sentinel (1),
letting s3dlio v0.9.112+ handle MPI-aware auto-sizing on its own
(via its `_pymod`'s `configure_thread_pools(0)` at import time).
"""

import os
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helper: partial-init ObjStoreLibStorage — bypass the full __init__
# (which would import s3dlio, run _preflight, etc.) and just wire the
# attributes _configure_s3dlio_runtime_env() reads.
# ---------------------------------------------------------------------------


def _partial_storage(write_threads):
    from dlio_benchmark.storage.obj_store_lib import ObjStoreLibStorage

    inst = ObjStoreLibStorage.__new__(ObjStoreLibStorage)
    inst._args = SimpleNamespace(write_threads=write_threads)
    return inst


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Each test starts with a clean S3DLIO_RT_THREADS / _S3DLIO_RT_AUTO."""
    monkeypatch.delenv("S3DLIO_RT_THREADS", raising=False)
    monkeypatch.delenv("_S3DLIO_RT_AUTO", raising=False)


# ---------------------------------------------------------------------------
# RED-then-GREEN regression tests
# ---------------------------------------------------------------------------


class TestS3dlioRuntimeEnvSentinelHandling:
    """The auto-size sentinel value of write_threads (=1) must NOT be
    propagated to S3DLIO_RT_THREADS.  Deriving RT threads from the
    sentinel poisons s3dlio's Tokio runtime with a 1-worker setup.
    """

    def test_sentinel_write_threads_does_not_set_s3dlio_rt_threads(self):
        """RED-then-GREEN core case: write_threads=1 (sentinel) →
        S3DLIO_RT_THREADS stays unset (s3dlio v0.9.112+ auto-sizes).

        Pre-fix, this test asserted `env["S3DLIO_RT_THREADS"] == "1"`
        against unmodified code; that same behavior is now what the
        test FORBIDS.  The RED-then-GREEN transition is bisectable
        against the same-PR fix commit.
        """
        inst = _partial_storage(write_threads=1)
        inst._configure_s3dlio_runtime_env()

        val = os.environ.get("S3DLIO_RT_THREADS")
        assert val is None, (
            f"ObjStoreLibStorage._configure_s3dlio_runtime_env() set "
            f"S3DLIO_RT_THREADS={val!r} even though write_threads is at "
            f"the auto-size sentinel (1).  This poisons s3dlio's Tokio "
            f"runtime with a 1-worker setup — every concurrent "
            f"multipart-upload part then serializes on it "
            f"(~10x throughput loss).  Expected: leave the env var "
            f"unset so s3dlio v0.9.112+'s MPI-aware default applies."
        )
        # And the _S3DLIO_RT_AUTO sentinel must not be set either —
        # nothing was auto-configured.
        assert "_S3DLIO_RT_AUTO" not in os.environ

    def test_explicit_write_threads_still_sets_s3dlio_rt_threads(self):
        """Regression guard for the intended path: if a user set
        write_threads > 1 explicitly in their YAML config (or DLIO
        auto-sized it BEFORE Storage init, which does not happen in
        the current flow but might in future refactors), the
        `write_threads * 1.5` (cap 128) derivation still applies —
        the fix is scoped strictly to the sentinel value.
        """
        inst = _partial_storage(write_threads=32)
        inst._configure_s3dlio_runtime_env()

        assert os.environ.get("S3DLIO_RT_THREADS") == "48", (
            "write_threads=32 must set S3DLIO_RT_THREADS=48 "
            "(1.5 × 32, cap 128).  Fix over-reached — auto-derive should "
            "still apply for real write_threads values."
        )
        assert os.environ.get("_S3DLIO_RT_AUTO") == "1"

    def test_user_preset_wins_over_auto_regardless_of_write_threads(self, monkeypatch):
        """User's explicit env-var setting (no _S3DLIO_RT_AUTO sentinel)
        must be preserved verbatim regardless of write_threads.
        """
        monkeypatch.setenv("S3DLIO_RT_THREADS", "7")
        inst = _partial_storage(write_threads=1)
        inst._configure_s3dlio_runtime_env()
        assert os.environ["S3DLIO_RT_THREADS"] == "7"

        inst2 = _partial_storage(write_threads=32)
        inst2._configure_s3dlio_runtime_env()
        assert os.environ["S3DLIO_RT_THREADS"] == "7"

    def test_re_derive_when_prior_ancestor_auto_set_and_write_threads_now_real(
        self, monkeypatch
    ):
        """Ancestor-recompute path (comment block preserved from the
        original design): if `_S3DLIO_RT_AUTO=1` is present, our
        earlier auto-set is stale and we should overwrite with the
        now-finalized `write_threads` value.
        """
        monkeypatch.setenv("S3DLIO_RT_THREADS", "1")  # stale ancestor value
        monkeypatch.setenv("_S3DLIO_RT_AUTO", "1")
        inst = _partial_storage(write_threads=32)
        inst._configure_s3dlio_runtime_env()
        assert os.environ["S3DLIO_RT_THREADS"] == "48"

    def test_ancestor_auto_set_but_write_threads_still_sentinel_leaves_stale(
        self, monkeypatch
    ):
        """Corner case: ancestor auto-set to 1, and this rank's
        write_threads is still at the sentinel too.  We CANNOT
        recompute (no info yet), so we skip — leaving the stale value
        would keep the poison alive, so we clear it back to unset
        instead, giving s3dlio's auto-sizing a chance to take effect.
        """
        monkeypatch.setenv("S3DLIO_RT_THREADS", "1")
        monkeypatch.setenv("_S3DLIO_RT_AUTO", "1")
        inst = _partial_storage(write_threads=1)
        inst._configure_s3dlio_runtime_env()
        # We explicitly clear the stale auto-set on the sentinel path
        # so s3dlio's own default takes effect.
        assert os.environ.get("S3DLIO_RT_THREADS") is None, (
            "Stale ancestor auto-set S3DLIO_RT_THREADS=1 should have "
            "been cleared when this rank has no better value either — "
            "leaving 1 keeps s3dlio's runtime crippled."
        )

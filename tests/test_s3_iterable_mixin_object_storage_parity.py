"""
Regression tests for object-storage library concurrency parity.

mlcommons/storage#626 bucket 2 — object-storage cross-library parity.

Background
==========
``_S3IterableMixin`` supports three storage libraries, chosen via
``storage_options.storage_library``: ``s3dlio``, ``minio``, and
``s3torchconnector``. Before this fix, the effective per-worker in-flight
I/O concurrency differed wildly across the three, even though a submitter
picks the library independently of the storage system under test:

  s3dlio            — ``s3dlio.get_many(max_in_flight=min(64, len(uris)))``
  minio             — ``ThreadPoolExecutor(max_workers=min(16, ...))``
                       (4x lower than s3dlio, undocumented)
  s3torchconnector  — fully sequential, one GET at a time (depth 1)

A submitter who happened to pick ``minio`` measured up to 4x less
concurrency than an ``s3dlio`` submitter on identical storage; a submitter
who picked ``s3torchconnector`` measured up to 64x less. Neither reflects
the storage system — only the client library choice. See the fairness
review referenced from storage#626 bucket 2.

Fix: all three libraries now share one ceiling,
``_MAX_PREFETCH_CONCURRENCY`` (64), via ``ThreadPoolExecutor`` fan-out for
minio and s3torchconnector (s3dlio already used a Rust-side max_in_flight
of 64 and is unchanged here).

Test strategy
=============
Concurrency claims are proven, not asserted from log lines: each test uses
a ``threading.Barrier`` that only releases once N callers have all
simultaneously entered a blocking I/O stand-in. If the code path is
actually running work N-at-a-time, the barrier releases and the test
passes quickly. If the code path only runs fewer than N at once (the
pre-fix minio cap, or pre-fix fully-sequential s3torchconnector), the
barrier can never be satisfied and ``barrier.wait()`` raises
``BrokenBarrierError`` once its timeout elapses — a fast, unambiguous RED.

``N=24`` is used for both: comfortably above the old minio ceiling (16,
so pre-fix minio genuinely cannot satisfy it) and comfortably below the
new ceiling (64, so post-fix minio/s3torchconnector satisfy it easily).
For s3torchconnector the ceiling itself doesn't matter for RED — ANY N>1
proves sequential-vs-concurrent, since pre-fix code never has more than
one ``.read()`` in flight.

Neither test touches a live S3 endpoint or requires ``minio`` to be
installed — both fake the boundary (``_get_minio_client`` /
``S3IterableDataset.from_objects``) with barrier-blocking stand-ins.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin


CONCURRENCY_PROBE_N = 24  # > old minio ceiling (16), < new ceiling (64)
BARRIER_TIMEOUT_S = 3.0  # generous for thread scheduling; RED fails fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(storage_library):
    """Bare mixin instance driven through the real _s3_init() so
    self._opts / self._storage_library / self._prefetch_pool etc. are all
    set up exactly as production code sets them. storage_library='s3dlio'
    is used at construction time deliberately (see _s3_init: 's3dlio' takes
    no third-party import) — the test then overrides _storage_library
    directly, bypassing the minio/s3torchconnector package-presence checks
    that _s3_init would otherwise enforce. This keeps the test independent
    of whether the minio package happens to be installed.
    """
    inst = _S3IterableMixin.__new__(_S3IterableMixin)
    inst._args = MagicMock()
    inst._args.storage_root = "test-bucket"
    inst._s3_init({"storage_library": "s3dlio"})
    inst._storage_library = storage_library
    return inst


class _BarrierBlockingReader:
    """Stand-in for a reader whose .read() only returns once
    CONCURRENCY_PROBE_N readers are all blocked in .read() at once."""

    def __init__(self, barrier):
        self._barrier = barrier

    def read(self):
        self._barrier.wait(timeout=BARRIER_TIMEOUT_S)
        return b"x" * 100


class _BarrierBlockingMinioResponse:
    def __init__(self, barrier):
        self._barrier = barrier

    def read(self):
        self._barrier.wait(timeout=BARRIER_TIMEOUT_S)
        return b"x" * 100

    def close(self):
        pass

    def release_conn(self):
        pass


class _BarrierBlockingMinioClient:
    def __init__(self, barrier):
        self._barrier = barrier

    def get_object(self, bucket, key):
        return _BarrierBlockingMinioResponse(self._barrier)


# ---------------------------------------------------------------------------
# MinIO — concurrency ceiling (RED against pre-fix code: capped at 16)
# ---------------------------------------------------------------------------


def test_prefetch_minio_runs_above_old_16_cap_concurrently():
    """storage#626 bucket 2: MinIO's ThreadPoolExecutor must run at least
    CONCURRENCY_PROBE_N (24) fetches concurrently — proving the ceiling is
    no longer hardcoded to 16.

    Pre-fix: ``n_workers = min(16, len(obj_keys))`` — with 24 objects
    requested, at most 16 threads ever call .read() at once. The barrier
    (which needs 24 simultaneous callers) can never be satisfied; every
    blocked thread's ``barrier.wait()`` raises ``BrokenBarrierError`` once
    BARRIER_TIMEOUT_S elapses, which propagates out of
    ``ThreadPoolExecutor.map()`` as the first raised exception seen by the
    caller.

    Post-fix: the ceiling is raised to match s3dlio/local (64), so all 24
    requested reads run concurrently and the barrier releases immediately.
    """
    barrier = threading.Barrier(CONCURRENCY_PROBE_N, timeout=BARRIER_TIMEOUT_S)
    inst = _make_instance("minio")
    inst._get_minio_client = lambda: _BarrierBlockingMinioClient(barrier)

    obj_keys = [f"obj-{i}" for i in range(CONCURRENCY_PROBE_N)]

    try:
        cache = inst._prefetch_minio(obj_keys)
    except threading.BrokenBarrierError:
        pytest.fail(
            "storage#626 bucket 2 regression: _prefetch_minio did not run "
            f"{CONCURRENCY_PROBE_N} fetches concurrently within "
            f"{BARRIER_TIMEOUT_S}s. MinIO's ThreadPoolExecutor is still "
            "capped below where s3dlio/local run (64) — a submitter using "
            "storage_library='minio' gets less concurrency than one using "
            "'s3dlio' on identical storage, purely from the library choice."
        )

    assert len(cache) == CONCURRENCY_PROBE_N
    assert all(v == 100 for v in cache.values())


# ---------------------------------------------------------------------------
# s3torchconnector — sequential vs concurrent (RED against pre-fix code)
# ---------------------------------------------------------------------------


def test_prefetch_s3torchconnector_runs_reads_concurrently():
    """storage#626 bucket 2: s3torchconnector reads must run concurrently,
    not one-at-a-time.

    Pre-fix: ``for obj_key, reader in zip(obj_keys, dataset): cache[obj_key]
    = len(reader.read())`` issues exactly one .read() at a time. With the
    barrier requiring CONCURRENCY_PROBE_N simultaneous callers, the single
    in-flight .read() blocks until BARRIER_TIMEOUT_S elapses, then raises
    BrokenBarrierError — a fast, unambiguous RED.

    Post-fix: readers are drained from the (network-free — see module
    docstring in _s3_iterable_mixin.py) sequential iterator first, then
    their .read() calls are fanned out across a ThreadPoolExecutor sized to
    the shared concurrency ceiling, matching s3dlio/minio.
    """
    barrier = threading.Barrier(CONCURRENCY_PROBE_N, timeout=BARRIER_TIMEOUT_S)
    fake_readers = [_BarrierBlockingReader(barrier) for _ in range(CONCURRENCY_PROBE_N)]
    fake_dataset = MagicMock()
    fake_dataset.__iter__.return_value = iter(fake_readers)

    inst = _make_instance("s3torchconnector")
    obj_keys = [f"obj-{i}" for i in range(CONCURRENCY_PROBE_N)]

    with patch(
        "s3torchconnector.S3IterableDataset.from_objects",
        return_value=fake_dataset,
    ):
        try:
            cache = inst._prefetch_s3torchconnector(obj_keys)
        except threading.BrokenBarrierError:
            pytest.fail(
                "storage#626 bucket 2 regression: _prefetch_s3torchconnector "
                f"did not run {CONCURRENCY_PROBE_N} reads concurrently within "
                f"{BARRIER_TIMEOUT_S}s. This path is still fully sequential "
                "(one GET in flight at a time) — a submitter using "
                "storage_library='s3torchconnector' gets up to 64x less "
                "concurrency than one using 's3dlio' on identical storage, "
                "purely from the library choice."
            )

    assert len(cache) == CONCURRENCY_PROBE_N
    assert all(v == 100 for v in cache.values())


# ---------------------------------------------------------------------------
# Structural check: single shared ceiling, not per-library magic numbers
# ---------------------------------------------------------------------------


def test_shared_prefetch_concurrency_constant_exists():
    """All three prefetch methods must derive their concurrency ceiling from
    one named module constant, not independently-chosen literals. This is
    what makes the storage#626 bucket 2 fix a structural guarantee rather
    than three numbers that happened to match on the day this landed and
    can silently drift apart again.
    """
    from dlio_benchmark.reader import _s3_iterable_mixin

    assert hasattr(_s3_iterable_mixin, "_MAX_PREFETCH_CONCURRENCY"), (
        "expected a single module-level _MAX_PREFETCH_CONCURRENCY constant "
        "shared by _prefetch_s3dlio, _prefetch_minio, and "
        "_prefetch_s3torchconnector — storage#626 bucket 2 parity would "
        "otherwise be three independently-maintained magic numbers."
    )
    assert _s3_iterable_mixin._MAX_PREFETCH_CONCURRENCY == 64

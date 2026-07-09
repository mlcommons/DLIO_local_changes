"""
Regression tests for fork-safety of the S3 iterable mixin's prefetch pool.

mlcommons/storage#626 bucket 1 (fork-safety hardening).

Background
==========
Before this fix, ``_s3_iterable_mixin`` defined a MODULE-LEVEL
``_PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, ...)`` at import time
in the parent process. When DLIO's DataLoader spawned workers via
``os.fork()``, the pool's worker thread did NOT survive fork — its state
(task queue, locks, work loop) was inherited by the child but the actual
OS-level thread was not. Any ``pool.submit()`` in a forked child would
enqueue work that nobody would drain, and ``future.result()`` would block
indefinitely.

The same hazard is what motivated the LOCAL_FS dispatcher gate in
``torch_data_loader.py`` (commit ``e4c9b7a``, fix for storage#391). The S3
mixin was safe in practice only because the s3dlio short-circuit at
``_s3_iterable_mixin.py:483`` bypasses the pool entirely — the minio and
s3torchconnector paths, which DO submit to the pool post-fork, were
latent.

Fix: the prefetch pool is created per-mixin-instance inside ``_s3_init()``,
which runs inside DLIO's ``worker_init()`` — strictly after ``os.fork()``.
Each child has its own live pool.

Test strategy
=============
1. ``test_prefetch_pool_survives_fork`` — behavioural: fork a real child
   process, exercise the mixin's prefetch pool, and require the child to
   complete a ``submit → result`` cycle within a short timeout. Pre-fix,
   the child hits the (dead-after-fork) module-level pool, blocks at
   ``future.result()``, and the parent has to terminate it — the test
   fails. Post-fix, each child creates its own pool in ``_s3_init()`` and
   the cycle completes.
2. ``test_no_module_level_prefetch_pool_in_s3_mixin`` — structural: locks
   the fix so a future refactor that reintroduces a module-level executor
   fails loudly instead of silently reintroducing the fork hazard.
"""
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest


# Import the module (not just the class) so we can introspect module-level
# state in the structural check. This mirrors DLIO's real startup: the
# parent process imports the mixin at benchmark setup, then forks workers.
from dlio_benchmark.reader import _s3_iterable_mixin
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_instance():
    """Bare mixin instance — same pattern as
    tests/test_direct_fs_iterable_mixin_gate.py. Bypasses ``__init__`` on
    purpose; the full reader stack pulls in mpi4py + torch + hydra, all
    irrelevant to the fork-safety property under test.

    Uses ``storage_library='s3dlio'`` because that path in ``_s3_init``
    performs no third-party imports — it only sets an env var. minio and
    s3torchconnector paths ``import`` their respective packages up front,
    which pollutes the test dependency surface without exercising anything
    the fork test needs.
    """
    inst = _S3IterableMixin.__new__(_S3IterableMixin)
    inst._args = MagicMock()
    inst._args.storage_options = {"storage_library": "s3dlio"}
    inst._args.storage_root = "test-bucket"
    return inst


def _child_body(result_queue):
    """Runs post-fork in the child process. Instantiate the mixin, drive
    its init path, then try to run a trivial task through whatever pool the
    mixin uses. Report success, failure, or an unexpected exception."""
    try:
        inst = _make_instance()
        inst._s3_init(inst._args.storage_options)

        # The observable property under test: the mixin's prefetch pool is
        # usable in this (forked) process. Prefer the per-instance
        # attribute added by the fix; fall back to the pre-fix module-level
        # pool so this test can go RED against unmodified code.
        pool = getattr(inst, "_prefetch_pool", None)
        if pool is None:
            pool = getattr(_s3_iterable_mixin, "_PREFETCH_POOL", None)
        if pool is None:
            result_queue.put(("no-pool", None))
            return

        future = pool.submit(lambda: "post-fork-alive")
        result = future.result(timeout=3.0)
        result_queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 — reporting for parent
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# Behavioural fork-safety test (RED against pre-fix code)
# ---------------------------------------------------------------------------

def test_prefetch_pool_survives_fork():
    """The S3 mixin's prefetch pool must be usable in a forked child.

    Pre-fix (module-level pool created at parent import time): the child
    inherits pool state whose worker thread never made it across fork.
    ``submit()`` enqueues into a queue nobody drains and ``result()``
    times out. This test then fails at either the ``p.is_alive()`` timeout
    guard or the ``status == 'ok'`` assertion below.

    Post-fix (per-instance pool created inside ``_s3_init()`` post-fork):
    the child has its own live pool and completes the cycle.

    Note on warming the parent pool: ``ThreadPoolExecutor`` starts its
    worker threads lazily on first ``submit()``. If the parent never
    submits before fork, no worker thread exists to lose, and the child's
    own first ``submit()`` spawns a worker in the child — hiding the bug.
    In real DLIO the parent doesn't submit before fork either, so the
    hazard is theoretical for lazily-created pools BUT becomes concrete
    the moment anything upstream (a preflight, a test harness, a health
    check) touches the pool in the parent. We force that warming here so
    the test faithfully reflects the worst-case fork-safety property we
    want the fix to guarantee — that no parent submission can taint the
    child, no matter what upstream does.
    """
    _pre_fix_pool = getattr(_s3_iterable_mixin, "_PREFETCH_POOL", None)
    if _pre_fix_pool is not None:
        _pre_fix_pool.submit(lambda: None).result(timeout=3.0)

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_child_body, args=(q,))
    proc.start()
    proc.join(timeout=10.0)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        pytest.fail(
            "storage#626 bucket 1 regression: the S3 mixin's prefetch pool "
            "hung in a forked child. The module-level ThreadPoolExecutor's "
            "worker thread does not survive os.fork(); .submit() enqueues "
            "work that nobody drains, and .result() blocks indefinitely. "
            "Fix: create the pool per-mixin-instance inside _s3_init() "
            "(runs post-fork inside DLIO's worker_init)."
        )

    assert not q.empty(), (
        "child did not report a result; likely crashed silently"
    )
    status, payload = q.get()
    assert status == "ok", (
        f"child failed to use the prefetch pool post-fork: status={status!r} "
        f"payload={payload!r} — this is the storage#626 bucket-1 hazard."
    )
    assert payload == "post-fork-alive", (
        f"unexpected payload from child: {payload!r}"
    )


# ---------------------------------------------------------------------------
# Structural check (RED against pre-fix code)
# ---------------------------------------------------------------------------

def test_no_module_level_prefetch_pool_in_s3_mixin():
    """The mixin module must not create a ThreadPoolExecutor at import
    time. Any executor built in the parent process before fork carries the
    storage#626 hazard: the child inherits state whose worker thread
    doesn't exist. Prefetch pools belong on the mixin instance, created in
    ``_s3_init()`` (post-fork).

    This is a structural lock on the fix — a future refactor that
    reintroduces module-level executor state fails here rather than
    silently reintroducing the hang.
    """
    offenders = [
        name for name, value in vars(_s3_iterable_mixin).items()
        if isinstance(value, ThreadPoolExecutor)
    ]
    assert not offenders, (
        f"module-level ThreadPoolExecutor(s) in _s3_iterable_mixin: "
        f"{offenders!r} — reintroduces storage#626 fork hazard. Prefetch "
        "pools must be created per-instance in _s3_init() so they live in "
        "the post-fork child, not the pre-fork parent."
    )

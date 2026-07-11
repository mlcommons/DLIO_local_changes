# S3-read resource-shape probe (DLIO v3.0.4, 2026-07-11)

**Status: DIAGNOSTIC ONLY.** This release does not fix
mlcommons/storage#755's root cause (that was on the reporter's side —
resource contention on a 16 GB / 8 CPU WSL VM co-located with MinIO).
It turns the reporter's opaque failure into a diagnosable one by
warning proactively about likely-infeasible workload shapes and by
annotating the failure message with a live system snapshot.

## Symptom that motivated this

From mlcommons/storage#755:

    RuntimeError: concurrent range chunk failed

... fired inside `_s3_iterable_mixin._s3_stream_s3dlio`'s `collect_batch`
loop, after four PyTorch DataLoader workers had all just reported they
were starting their s3dlio sliding window. No indication in the
Python traceback of what actually went wrong on the wire.

Environment details later provided by the reporter: WSL VM with 16 GB
of RAM and at most 8 CPUs, running the MinIO S3 target on the same
host. Every s3dlio Tokio worker, every DataLoader worker process, and
the MinIO server itself were competing for the same 8 cores. The
configured workload (`prefetch_window=64` × 4 DataLoader workers ×
146 MB unet3d records) implied 37 GiB of in-flight buffer memory
before the sliding window was full — 2.3× the VM's total RAM.

The root cause on the reporter's end is a workload sizing / colocation
issue — this repo cannot fix it. But the message they got gave them
nothing to work with.

## What this release adds

`dlio_benchmark/utils/resource_probe.py` — two entry points, both
non-raising by design (a broken /proc read must not mask the real
error):

1. `check_workload_resources(*, prefetch, dataloader_workers,
   record_length_bytes, ranks_per_node)` — returns a list of warning
   strings.  Three checks against the current process's observable
   limits:

   - **FD exhaustion**: `prefetch × dataloader_workers × ranks_per_node × 2`
     projected fds vs `ulimit -n` soft limit.
   - **RAM exhaustion**: `prefetch × dataloader_workers × ranks × record_length`
     projected in-flight buffer memory vs `MemAvailable / 2` from
     `/proc/meminfo`.
   - **CPU oversubscription**: `(tokio_workers + dataloader_workers) ×
     ranks_per_node` projected total contending threads vs `cpu_count × 4`
     (the threshold beyond which the OS scheduler starts thrashing under
     load and any co-located server starves).

2. `augment_error_with_snapshot(exc, context)` — returns a new
   `RuntimeError` chained (via `raise ... from exc`) to the original,
   with a one-line resource snapshot (fd count, RSS, load-avg)
   captured at the moment of failure appended.  The original stays
   visible as `__cause__` in the traceback.

Both entry points are wired into
`_s3_iterable_mixin._s3_stream_s3dlio()`:

- Right before the `collect_batch` loop: `emit_workload_warnings(...)`
  fires once per DataLoader worker with the reader's own `prefetch`,
  `args.read_threads`, `args.record_length`, and
  `DLIOMPI.ranks_per_node()`.
- The loop is wrapped in `try: ... except RuntimeError as exc:` — on
  failure, `augment_error_with_snapshot` re-raises with the live
  snapshot appended.

Everything is wrapped in defensive `try:` so the probe itself never
short-circuits a working read.

## What the reporter would see now

At read start, before the first `collect_batch` call:

    WARNING NPZReaderS3Iterable[thread=0] [resource-probe] RAM projection:
    prefetch=64 x dataloader_workers=4 x ranks_per_node=1 x
    record_length=146 MiB = 37376 MiB of in-flight buffer memory, but
    only 12000 MiB (of 16000 MiB total) is currently available.
    OOM / swap-thrash under load is likely.  Fix: reduce
    storage_options.prefetch_window, or reduce DataLoader worker count,
    or move the S3 target off-node.

If they ignored the warning and the loop tripped anyway:

    RuntimeError: NPZReaderS3Iterable._s3_stream_s3dlio [thread=0,
    prefetch=64, collect_n=7, uris=1800, skip_head=False]:
    RuntimeError: concurrent range chunk failed
      Resource state at time of failure: fd=980/1024 (hard 1048576),
      RSS=15200 MiB / 200 MiB avail / 16000 MiB total, cpu=8, load1=42.5
      Run with logging.WARNING to see storage-side resource-check
      projections; see dlio_benchmark/utils/resource_probe.py.

    Caused by: RuntimeError: concurrent range chunk failed: <s3dlio's
    v0.9.112 preserved anyhow cause chain here>

Both messages point at "your buffer memory demand exceeds available
RAM" as the load-bearing signal, and the suggested fix is workload-
side (`prefetch_window: 4` or fewer workers), not client-side.

## Test coverage

`tests/test_resource_probe.py` — 14 tests:

- `TestSnapshot`: `snapshot()` returns a sane `ResourceSnapshot` on
  the current host; `one_line()` is a formatted string.
- `TestFdWarning`, `TestRamWarning`, `TestCpuWarning`: each warning
  fires on its designed trigger shape and stays silent on a comfortable
  shape.  Sentinel case `record_length_bytes=0` skips the RAM check.
- `TestReporterShape`: the exact WSL config the reporter ran
  (16 GB / 8 CPU, prefetch=64, workers=4, records=146 MB, NP=1)
  MUST fire the RAM warning.
- `TestEmitWorkloadWarnings`: the emitter logs each warning at
  WARNING level with the caller-supplied context prefix (verified via
  `caplog`).
- `TestAugmentErrorWithSnapshot`: the wrapped error contains the
  original exception's type name and text plus the snapshot; the
  augmenter never masks the original error even if the snapshot
  itself raises (`monkeypatch` a broken `snapshot`).

All 115 tests pass across the resource-probe, minio-adapter, sentinel,
and fast_ci suites; `ruff check` and `ruff format --check` clean on
every file touched.

## What this does NOT do

- Does not change any failure/success semantics of an S3 op — a run
  that would have failed still fails.
- Does not tune s3dlio's Tokio runtime, connection pool, or in-flight
  concurrency.  Those knobs stay under user control via
  `storage_options.prefetch_window` / `S3DLIO_MAX_CONCURRENCY` /
  `S3DLIO_POOL_MAX_IDLE_PER_HOST`.
- Does not attempt to auto-scale `prefetch_window` down when RAM looks
  tight — that's a policy choice that belongs on the maintainer side.
- Does not probe the S3 endpoint itself.  Endpoint-side pressure
  (MinIO CPU throttling, network path RTT) is invisible from inside
  the DLIO process; if you want that signal, monitor MinIO's own
  metrics endpoint alongside the run.

## Cross-repo references

- s3dlio error-chain preservation (which lets the underlying cause
  actually reach Python now): [`../../s3dlio/docs/Changelog.md`](../../s3dlio/docs/Changelog.md) v0.9.112
- s3dlio `concurrent_range_get_impl` (source of the wrapper message):
  [`../../s3dlio/src/s3_utils.rs`](../../s3dlio/src/s3_utils.rs) around the
  "concurrent range chunk failed" `context()` call

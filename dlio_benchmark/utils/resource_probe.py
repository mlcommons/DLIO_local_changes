"""System-resource sanity checks for object-storage workloads.

Motivating case: mlcommons/storage#755.  The reporter's environment was
a WSL VM with 16 GB RAM and at most 8 CPUs, running a co-located MinIO
server as the S3 target.  Every S3 client operation, every s3dlio
Tokio worker, every DataLoader worker process, AND the MinIO server
itself were competing for the same 8 cores and 16 GB.  The failing
symptom was:

    RuntimeError: concurrent range chunk failed

... which is s3dlio's generic wrapper around the first task-level
error inside `concurrent_range_get_impl` — the underlying cause
(SDK error, timeout, TLS blip, 5xx from an overloaded MinIO, TCP
reset from ephemeral-port exhaustion, ...) is preserved in the
`anyhow` chain in s3dlio v0.9.112+, but even a well-formed error
message doesn't help a user who does not yet realize their MinIO
server is falling over under co-location pressure.

This module makes the LIKELY causes visible up-front by comparing
the configured workload shape (in-flight prefetch, DataLoader-worker
count, record size) against actual observable system limits
(``ulimit -n``, physical/available RAM, CPU count).  Two entry
points cover the two useful moments:

1. `check_workload_resources(...)` — pure, returns a list of
   warning strings.  Called from `_s3_iterable_mixin._s3_stream_s3dlio`
   before the read loop starts, once per DataLoader worker: warnings
   fire proactively at workload start, well before the failure would
   otherwise surface.

2. `augment_error_with_snapshot(exc, context)` — takes a caught
   exception, samples the LIVE resource state, and returns a new
   `RuntimeError` chained (via `raise ... from exc`) to the original
   with a compact system snapshot appended.  Called from the reader's
   `collect_batch` except handler: the traceback now shows both the
   original error AND the resource state at the moment it fired,
   letting the reporter/reviewer see fd exhaustion / RSS-near-limit /
   load-average-way-over-CPU directly.

Neither function changes the failure/success semantics of any
operation — they add diagnostic hints only.  A `try:` around the
snapshot calls means resource-probing itself never masks the real
error.
"""

from __future__ import annotations

import logging
import os
import resource
from dataclasses import dataclass
from typing import Optional


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time system resource snapshot for this process."""

    #: Soft file-descriptor limit (`ulimit -n` in bash).
    fd_soft: int
    #: Hard file-descriptor limit.
    fd_hard: int
    #: Currently open file descriptors on this process (from ``/proc/self/fd``).
    fd_open: int
    #: Resident-set size in MiB (from ``/proc/self/status``, VmRSS).
    rss_mib: int
    #: MemAvailable in MiB (from ``/proc/meminfo``).
    mem_avail_mib: int
    #: MemTotal in MiB (from ``/proc/meminfo``).
    mem_total_mib: int
    #: Number of usable CPUs (``os.sched_getaffinity`` on Linux, fallback to
    #: ``os.cpu_count()``).
    cpu_count: int
    #: 1-minute load average (from ``os.getloadavg``).
    load_1min: float

    def one_line(self) -> str:
        """Compact, single-line human-readable rendering for log lines."""
        return (
            f"fd={self.fd_open}/{self.fd_soft} (hard {self.fd_hard}), "
            f"RSS={self.rss_mib} MiB / {self.mem_avail_mib} MiB avail "
            f"/ {self.mem_total_mib} MiB total, "
            f"cpu={self.cpu_count}, load1={self.load_1min:.2f}"
        )


def _read_proc_status_kib(field: str) -> int:
    """Return an integer KiB field from ``/proc/self/status`` (e.g. VmRSS)."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    parts = line.split()
                    # "VmRSS:\t12345 kB"
                    return int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _read_proc_meminfo_kib(field: str) -> int:
    """Return an integer KiB field from ``/proc/meminfo`` (e.g. MemAvailable)."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    parts = line.split()
                    return int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _count_open_fds() -> int:
    """Count entries in ``/proc/self/fd``.  Fast on Linux, returns 0 on error."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def snapshot() -> ResourceSnapshot:
    """Sample the current process's resource state.

    All fields default to 0 on read error rather than raising, so this
    function is safe to call from any code path — including from inside
    an ``except`` handler where raising would mask the real error.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        soft, hard = 0, 0

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 0

    try:
        load_1min = os.getloadavg()[0]
    except OSError:
        load_1min = 0.0

    return ResourceSnapshot(
        fd_soft=soft,
        fd_hard=hard,
        fd_open=_count_open_fds(),
        rss_mib=_read_proc_status_kib("VmRSS") // 1024,
        mem_avail_mib=_read_proc_meminfo_kib("MemAvailable") // 1024,
        mem_total_mib=_read_proc_meminfo_kib("MemTotal") // 1024,
        cpu_count=cpu_count,
        load_1min=load_1min,
    )


def check_workload_resources(
    *,
    prefetch: int,
    dataloader_workers: int,
    record_length_bytes: int,
    ranks_per_node: int = 1,
    snap: Optional[ResourceSnapshot] = None,
) -> list[str]:
    """Return warning strings for an S3 iterable-read workload shape that
    LOOKS likely to exhaust a system resource, given the current process's
    observable limits.

    The three checks correspond to the three classes of failure that
    caused #755-style opacity on constrained co-located systems:

    - **FD exhaustion**: HTTP/1.1 keepalive sockets are one fd each.
      `prefetch × dataloader_workers` GETs may all be in flight
      simultaneously; on a low `ulimit -n` VM this can hit the process
      cap and manifest as a mid-run reset.  Multiply by 2 for headroom
      (Python/libc's own fds, MPI, dftracer, etc.).

    - **RAM exhaustion**: s3dlio's sliding-window prefetch holds up to
      `prefetch` complete objects in memory per DataLoader worker.
      For a 146 MB record and `prefetch=64 × workers=4`, that's ~37 GiB
      just for in-flight buffers, which doesn't fit on the 16 GiB WSL
      VM the reporter of #755 was on.

    - **CPU oversubscription**: Total logical concurrency = Tokio worker
      count × DataLoader workers × MPI ranks-per-node.  When this
      exceeds physical CPUs by more than ~4× the OS starts thrashing
      under load, MinIO (or any co-located server) starves, and the
      SDK-layer requests time out.

    Args:
        prefetch: s3dlio in-flight GETs per DataLoader worker
            (``storage_options.prefetch_window``, default 64).
        dataloader_workers: PyTorch/TF DataLoader worker processes.
        record_length_bytes: Expected size of each object.  Approximate
            is fine — the check uses order-of-magnitude thresholds.
        ranks_per_node: MPI ranks sharing this node (default 1).
        snap: Optional pre-taken snapshot (for tests + to avoid a second
            syscall round-trip in the caller); pass ``None`` in prod.

    Returns:
        A list of warning strings.  Empty if nothing looks off.
        Callers typically `logging.warning()` each one at storage / read
        start; nothing is raised.
    """
    if snap is None:
        snap = snapshot()
    warnings: list[str] = []

    # 1. FD headroom.
    projected_sockets = prefetch * max(1, dataloader_workers) * max(1, ranks_per_node)
    projected_fds = projected_sockets * 2  # 2x headroom for libc/mpi/etc.
    if snap.fd_soft > 0 and projected_fds > snap.fd_soft:
        warnings.append(
            f"[resource-probe] fd projection: prefetch={prefetch} x "
            f"dataloader_workers={dataloader_workers} x ranks_per_node="
            f"{ranks_per_node} = {projected_sockets} concurrent sockets "
            f"(x2 headroom = {projected_fds} fds), but this process's "
            f"soft fd limit is only {snap.fd_soft} "
            f"(hard {snap.fd_hard}).  Mid-run failures with "
            f"'Too many open files' / 'concurrent range chunk failed' / "
            f"connection resets are likely.  Fix: `ulimit -n {snap.fd_hard}` "
            f"before launching, or reduce storage_options.prefetch_window."
        )

    # 2. RAM headroom for in-flight buffer memory.
    if record_length_bytes > 0 and snap.mem_avail_mib > 0:
        buffer_mib = (
            prefetch
            * max(1, dataloader_workers)
            * max(1, ranks_per_node)
            * record_length_bytes
            // (1024 * 1024)
        )
        # A worry-worthy threshold is 50% of currently-available RAM —
        # we don't want to be the reason the OOM killer wakes up.
        if buffer_mib > snap.mem_avail_mib // 2:
            warnings.append(
                f"[resource-probe] RAM projection: prefetch={prefetch} x "
                f"dataloader_workers={dataloader_workers} x "
                f"ranks_per_node={ranks_per_node} x "
                f"record_length={record_length_bytes // (1024 * 1024)} MiB "
                f"= {buffer_mib} MiB of in-flight buffer memory, but only "
                f"{snap.mem_avail_mib} MiB (of {snap.mem_total_mib} MiB total) "
                f"is currently available.  OOM / swap-thrash under load "
                f"is likely.  Fix: reduce storage_options.prefetch_window, "
                f"or reduce DataLoader worker count, or move the S3 target "
                f"off-node."
            )

    # 3. CPU oversubscription.
    # Tokio worker count on this rank: max(4, cpu_count/ranks_per_node) by default.
    tokio_workers = (
        max(4, snap.cpu_count // max(1, ranks_per_node)) if snap.cpu_count else 0
    )
    total_threads = (tokio_workers + max(1, dataloader_workers)) * max(
        1, ranks_per_node
    )
    if snap.cpu_count > 0 and total_threads > snap.cpu_count * 4:
        warnings.append(
            f"[resource-probe] CPU projection: tokio_workers={tokio_workers} + "
            f"dataloader_workers={dataloader_workers} per rank, "
            f"ranks_per_node={ranks_per_node} = {total_threads} total "
            f"contending threads, but only {snap.cpu_count} logical CPUs.  "
            f"Ratio {total_threads / snap.cpu_count:.1f}x is well above the "
            f"~4x threshold where the scheduler starts thrashing.  If a "
            f"co-located S3 server (MinIO, etc.) is competing for these same "
            f"CPUs, request timeouts / 'concurrent range chunk failed' are "
            f"likely under load.  Fix: reduce dataloader_workers, reduce "
            f"prefetch, or move the S3 target off-node."
        )

    return warnings


def emit_workload_warnings(
    context: str,
    *,
    prefetch: int,
    dataloader_workers: int,
    record_length_bytes: int,
    ranks_per_node: int = 1,
) -> None:
    """Convenience: run `check_workload_resources` and `LOG.warning` each result.

    ``context`` is prefixed to each warning so operators can tell WHICH
    reader / rank emitted the finding.
    """
    for w in check_workload_resources(
        prefetch=prefetch,
        dataloader_workers=dataloader_workers,
        record_length_bytes=record_length_bytes,
        ranks_per_node=ranks_per_node,
    ):
        LOG.warning("%s %s", context, w)


def augment_error_with_snapshot(
    exc: BaseException,
    context: str,
) -> RuntimeError:
    """Wrap ``exc`` with a fresh resource snapshot for the caller to
    re-raise.

    Usage::

        try:
            ...
        except RuntimeError as exc:
            raise augment_error_with_snapshot(
                exc, "NPZReaderS3Iterable._s3_stream_s3dlio"
            ) from exc

    The returned `RuntimeError` message includes the original exception
    text AND the one-line resource snapshot at the moment of failure.
    The caller MUST use ``raise ... from exc`` so the original exception
    stays visible as `__cause__` in the traceback.

    Never raises from the snapshot itself — a broken /proc read must
    not mask the real error being reported.
    """
    try:
        snap_str = snapshot().one_line()
    except BaseException:  # noqa: BLE001 — must not mask
        snap_str = "<resource snapshot failed>"

    return RuntimeError(
        f"{context}: {type(exc).__name__}: {exc}\n"
        f"  Resource state at time of failure: {snap_str}\n"
        f"  Run with logging.WARNING to see storage-side resource-check "
        f"projections; see dlio_benchmark/utils/resource_probe.py."
    )

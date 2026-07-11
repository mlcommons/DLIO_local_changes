"""Tests for the workload resource-probe helper (see storage#755).

The helper is diagnostics-only — no test verifies a *fix* to #755's root
cause (that was on the reporter's side, resource contention on a
16 GB / 8 CPU WSL VM co-located with MinIO).  These tests verify that
DLIO now proactively flags the LIKELY causes at storage-init /
read-start time and annotates the failure message with a live snapshot
if the read loop DOES fall over.
"""

from dlio_benchmark.utils import resource_probe as rp


def _snap(**overrides) -> rp.ResourceSnapshot:
    """Build a ResourceSnapshot with sensible defaults for tests."""
    defaults = dict(
        fd_soft=1024,
        fd_hard=1_048_576,
        fd_open=200,
        rss_mib=500,
        mem_avail_mib=12_000,
        mem_total_mib=16_000,
        cpu_count=8,
        load_1min=1.5,
    )
    defaults.update(overrides)
    return rp.ResourceSnapshot(**defaults)


class TestSnapshot:
    """The real snapshot() call must return a sane ResourceSnapshot on
    the current process — the values don't matter for the test, only
    that no exception fires and the fields have plausible ranges.
    """

    def test_snapshot_returns_populated_dataclass(self):
        snap = rp.snapshot()
        assert isinstance(snap, rp.ResourceSnapshot)
        assert snap.fd_soft > 0
        assert snap.fd_hard >= snap.fd_soft
        assert snap.fd_open >= 3  # at least stdin/stdout/stderr
        assert snap.cpu_count >= 1
        assert snap.mem_total_mib > 0
        assert snap.mem_avail_mib <= snap.mem_total_mib

    def test_one_line_is_a_string(self):
        snap = rp.snapshot()
        s = snap.one_line()
        assert isinstance(s, str)
        # Sanity check that all four sections are present.
        for token in ("fd=", "RSS=", "cpu=", "load1="):
            assert token in s


class TestFdWarning:
    """Warn when projected socket count exceeds soft fd limit."""

    def test_fd_warning_fires_on_low_soft_limit(self):
        # Reporter's WSL default: soft ulimit ~1024.  prefetch=64, workers=4,
        # ranks=2 => 512 sockets x2 headroom = 1024 fds — exactly at limit.
        # Push slightly higher to trip the check.
        snap = _snap(fd_soft=1024, fd_hard=1_048_576)
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=1024,  # tiny, so RAM check doesn't fire
            ranks_per_node=3,
            snap=snap,
        )
        fd_warnings = [w for w in warnings if "fd projection" in w]
        assert len(fd_warnings) == 1
        # The message must be actionable: name the specific ulimit hint.
        assert "ulimit -n" in fd_warnings[0]
        assert "prefetch_window" in fd_warnings[0]

    def test_fd_warning_silent_when_headroom_ok(self):
        # Reasonable soft limit + typical workload.
        snap = _snap(fd_soft=65536)
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=1024,
            ranks_per_node=1,
            snap=snap,
        )
        assert not any("fd projection" in w for w in warnings)


class TestRamWarning:
    """Warn when projected in-flight buffer memory dwarfs available RAM."""

    def test_ram_warning_fires_on_large_record_and_high_concurrency(self):
        # Reporter's shape: 146 MB records, 64 in-flight, 4 workers on a
        # 16 GB WSL => 37 GiB projected buffer memory, way over 12 GiB avail.
        snap = _snap(mem_avail_mib=12_000, mem_total_mib=16_000)
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=146 * 1024 * 1024,
            ranks_per_node=1,
            snap=snap,
        )
        ram_warnings = [w for w in warnings if "RAM projection" in w]
        assert len(ram_warnings) == 1
        assert "prefetch_window" in ram_warnings[0]

    def test_ram_warning_silent_when_records_are_small(self):
        # JPEG-sized workload — even prefetch=64 * workers=4 is trivial.
        snap = _snap(mem_avail_mib=12_000)
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=150 * 1024,
            ranks_per_node=1,
            snap=snap,
        )
        assert not any("RAM projection" in w for w in warnings)

    def test_ram_warning_silent_when_record_length_unknown(self):
        # record_length_bytes=0 is the "we don't know" sentinel — skip
        # the check rather than warn spuriously.
        snap = _snap(mem_avail_mib=100)  # very low
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=0,
            ranks_per_node=1,
            snap=snap,
        )
        assert not any("RAM projection" in w for w in warnings)


class TestCpuWarning:
    """Warn when total contending threads dwarf physical CPUs."""

    def test_cpu_warning_fires_on_oversubscribed_small_vm(self):
        # 8-CPU WSL, 4 dataloader workers per rank, 2 ranks/node.
        # Tokio workers = max(4, 8/2) = 4; total = (4+4)*2 = 16
        # threads (< 8*4 = 32 threshold, so this should NOT fire).
        # Bump to 8 ranks to trip: (max(4, 8/8)=4 + 4) * 8 = 64 > 32.
        snap = _snap(cpu_count=8)
        warnings = rp.check_workload_resources(
            prefetch=1,  # keep fd + RAM checks silent
            dataloader_workers=4,
            record_length_bytes=1024,
            ranks_per_node=8,
            snap=snap,
        )
        cpu_warnings = [w for w in warnings if "CPU projection" in w]
        assert len(cpu_warnings) == 1
        assert "co-located S3 server" in cpu_warnings[0]

    def test_cpu_warning_silent_on_beefy_host(self):
        snap = _snap(cpu_count=64)
        warnings = rp.check_workload_resources(
            prefetch=1,
            dataloader_workers=4,
            record_length_bytes=1024,
            ranks_per_node=1,
            snap=snap,
        )
        assert not any("CPU projection" in w for w in warnings)


class TestReporterShape:
    """The exact reporter's config in mlcommons/storage#755 must fire at
    least the RAM warning — that's the strongest signal on their env.
    """

    def test_reporter_shape_fires_ram_warning(self):
        # Reporter's approximate env: 16 GB RAM WSL, 8 CPU, 4 DataLoader
        # workers, 64 in-flight, 146 MB records, NP=1.
        snap = _snap(
            fd_soft=1024,
            mem_avail_mib=12_000,
            mem_total_mib=16_000,
            cpu_count=8,
        )
        warnings = rp.check_workload_resources(
            prefetch=64,
            dataloader_workers=4,
            record_length_bytes=146 * 1024 * 1024,
            ranks_per_node=1,
            snap=snap,
        )
        assert any("RAM projection" in w for w in warnings), (
            "The reporter's 4x64x146MB config = 37 GiB in-flight buffer "
            "memory MUST fire the RAM projection warning on their 16 GiB "
            "VM.  Without this, users on constrained systems get an "
            "opaque 'concurrent range chunk failed' with no hint that "
            "the FIRST thing to try is `prefetch_window: 4`."
        )


class TestEmitWorkloadWarnings:
    """The convenience emitter logs each warning at WARNING level with a
    caller-supplied context prefix so operators can tell which reader
    fired.  Uses `caplog` to verify.
    """

    def test_emit_calls_logging_warning_per_check_result(self, caplog):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger=rp.LOG.name):
            rp.emit_workload_warnings(
                "NPZReaderS3Iterable[rank=0,thread=0]",
                prefetch=64,
                dataloader_workers=4,
                record_length_bytes=146 * 1024 * 1024,
                ranks_per_node=1,
            )

        warning_records = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert warning_records, "expected at least one WARNING to be logged"
        for r in warning_records:
            assert "NPZReaderS3Iterable[rank=0,thread=0]" in r.getMessage()


class TestAugmentErrorWithSnapshot:
    """The error-augmenter takes an exception, samples current state,
    and returns a new RuntimeError containing both.  The original
    exception must remain visible as `__cause__` when the caller does
    `raise ... from exc`.
    """

    def test_augmented_error_message_includes_original_and_snapshot(self):
        try:
            raise RuntimeError("concurrent range chunk failed: transient error")
        except RuntimeError as exc:
            wrapped = rp.augment_error_with_snapshot(exc, "some.reader.context")

        msg = str(wrapped)
        assert "concurrent range chunk failed" in msg
        assert "some.reader.context" in msg
        # Snapshot section is present.
        for token in ("fd=", "RSS=", "cpu=", "load1="):
            assert token in msg

    def test_augmented_error_returned_object_is_runtimeerror(self):
        wrapped = rp.augment_error_with_snapshot(ValueError("boom"), "ctx")
        assert isinstance(wrapped, RuntimeError)
        # And carries the ORIGINAL type name in the message so the
        # reviewer knows what to search for.
        assert "ValueError" in str(wrapped)

    def test_augmenter_never_masks_original_error(self, monkeypatch):
        """If /proc read fails (containerized weird env, etc.), the
        wrapped error must still surface the original.
        """

        def _boom() -> rp.ResourceSnapshot:
            raise OSError("simulated /proc read failure")

        monkeypatch.setattr(rp, "snapshot", _boom)

        wrapped = rp.augment_error_with_snapshot(
            RuntimeError("concurrent range chunk failed"), "ctx"
        )
        assert "concurrent range chunk failed" in str(wrapped)
        # The snapshot section is present but empty/errored.
        assert "resource snapshot failed" in str(wrapped)

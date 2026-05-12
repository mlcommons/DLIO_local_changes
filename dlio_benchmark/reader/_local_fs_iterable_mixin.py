"""
_LocalFSIterableMixin — parallel prefetch for local-filesystem iterable readers.

WHY THIS EXISTS — PARITY WITH _S3IterableMixin
===============================================
DLIO is a storage benchmark. FormatReader.next() always yields
``self._args.resized_image`` — a single pre-allocated dummy tensor. The actual
decoded file bytes are NEVER used. They are consulted for exactly one thing:
the ``image_size`` metric inside ``dlp.update(image_size=N)``.

Without this mixin, local-FS readers open and decode files ONE AT A TIME inside
the next() loop (queue depth = 1). The S3 iterable readers pre-fetch ALL files
in parallel before the iteration starts (queue depth = N). This is a structural
parity violation — local-FS benchmarks look slower than they physically should
be, making cross-backend comparisons invalid.

This mixin gives local-FS readers the same pre-fetch pattern as _S3IterableMixin:

1. Before next(): parallel-read all assigned files via ThreadPoolExecutor (buffered)
   OR via s3dlio.get_many() with direct:// URIs (O_DIRECT, page-cache bypass).
2. Store only the raw byte count per file (never decode numpy/PIL/h5py)
3. During next() / get_sample(): dict lookup → telemetry → return resized_image

I/O IS FULLY MEASURED
=====================
The full read() of each file still happens inside _localfs_prefetch_all().
Only the decode step (np.load, PIL.open, h5py.File) is skipped — that decode
is pure CPU overhead that has nothing to do with storage bandwidth.

TWO PREFETCH MODES
==================
storage_library: <unset or "posix">
    ThreadPoolExecutor(64) + Python open() + buffered read.
    Simple, portable, uses OS page cache.

storage_library: "direct"
    s3dlio.get_many() with direct:// URIs.
    Uses O_DIRECT (Linux) — bypasses page cache entirely, 4 KiB-aligned I/O
    via Tokio async tasks in the s3dlio Rust runtime. GIL is released for the
    full duration of all reads.
    **Required for accurate NVMe benchmarking** — repeated buffered reads hit
    the page cache rather than the device, understating storage latency and
    saturating DRAM bandwidth instead of device bandwidth.

USAGE PATTERN
=============
Subclass from BOTH the format-specific parent AND this mixin::

    class ImageReader(_OriginalImageReader, _LocalFSIterableMixin):
        @dlp.log_init
        def __init__(self, dataset_type, thread_index, epoch):
            super().__init__(dataset_type, thread_index, epoch)
            self._localfs_init()

        @dlp.log
        def open(self, filename):
            return self._local_cache.get(filename, 0)

        @dlp.log
        def get_sample(self, filename, sample_index):
            dlp.update(image_size=self._local_cache.get(filename, 0))

        def next(self):
            self._localfs_prefetch_all()
            for batch in super().next():
                yield batch
"""
import os
from concurrent.futures import ThreadPoolExecutor

from dlio_benchmark.utils.utility import utcnow


class _LocalFSIterableMixin:
    """
    Mixin providing parallel local-filesystem prefetch for iterable readers.

    Do NOT instantiate directly. Mix in alongside a FormatReader subclass;
    call ``_localfs_init()`` from the subclass ``__init__`` after
    ``super().__init__()``.

    Set ``storage_library: direct`` in storage_options to use s3dlio's O_DIRECT
    path (bypasses page cache — essential for accurate NVMe benchmarking).
    Default (no storage_library, or ``posix``) uses buffered Python open().
    """

    def _localfs_init(self) -> None:
        """
        Initialise mixin state.

        Reads ``storage_options.storage_library`` from ConfigArguments:
          - ``"direct"`` → s3dlio O_DIRECT path (``direct://`` URIs, Tokio, GIL-free)
          - anything else → buffered Python ThreadPoolExecutor path

        Sets:
          - ``self._local_cache``      (dict: filename → int byte count)
          - ``self._use_direct``       (bool)
          - ``self._storage_root``     (str absolute path, for direct:// URI construction)
          - ``self._total_bytes_read`` (int, epoch accumulator)
          - ``self._total_objects_read`` (int, epoch accumulator)
        """
        self._local_cache: dict = {}
        self._total_bytes_read: int = 0
        self._total_objects_read: int = 0

        opts = getattr(self._args, "storage_options", {}) or {}
        lib = opts.get("storage_library", "")
        self._use_direct: bool = (lib == "direct")

        if self._use_direct:
            try:
                import s3dlio as _s3dlio  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    f"{self.__class__.__name__}: storage_library='direct' requires "
                    "the s3dlio package. Install with: pip install s3dlio"
                ) from exc

    # ── URI helpers ───────────────────────────────────────────────────────────

    def _direct_uri_for_path(self, path: str) -> str:
        """Return a ``direct://`` URI for an absolute or relative local path."""
        return f"direct://{os.path.abspath(path)}"

    # ── Buffered path (default) ───────────────────────────────────────────────

    def _read_local_bytes(self, path: str) -> int:
        """Read a local file using buffered I/O and return its byte count. No decode."""
        with open(path, 'rb') as fh:
            return len(fh.read())

    def _prefetch_buffered(self, paths: list) -> dict:
        """
        Parallel buffered reads via ThreadPoolExecutor(64).

        Uses the OS page cache. Fast for warm-cache runs; not representative of
        cold-device bandwidth on NVMe.
        """
        n_workers = min(64, len(paths))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for path, byte_count in zip(paths, pool.map(self._read_local_bytes, paths)):
                cache[path] = byte_count
        return cache

    # ── O_DIRECT path (storage_library: direct) ───────────────────────────────

    def _prefetch_direct(self, paths: list) -> dict:
        """
        Parallel O_DIRECT reads via ``s3dlio.get_many()`` with ``direct://`` URIs.

        - Bypasses the OS page cache (Linux O_DIRECT, 4 KiB-aligned buffers).
        - Runs in Tokio async tasks inside the s3dlio Rust runtime; GIL is
          released for the full duration.
        - ``len(data)`` is O(1) on the returned BytesView — no Python bytes copy.
        - Up to 64 concurrent reads in flight (same as _prefetch_buffered workers).

        This is the correct mode for NVMe benchmarks: it stresses the device
        itself rather than DRAM bandwidth or page-cache eviction policy.
        """
        import s3dlio

        uris = [self._direct_uri_for_path(p) for p in paths]
        uri_to_path = dict(zip(uris, paths))
        max_in_flight = min(64, len(uris))
        results = s3dlio.get_many(uris, max_in_flight=max_in_flight)

        cache = {}
        for uri, data in results:
            path = uri_to_path.get(uri, uri)
            cache[path] = len(data)   # byte count only; BytesView.len() is O(1)
        return cache

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _localfs_prefetch_all(self) -> None:
        """
        Collect all files assigned to this thread and prefetch them in parallel.

        Routes to the O_DIRECT (s3dlio direct://) or buffered (Python open())
        path based on the ``storage_library`` setting from _localfs_init().

        Call at the top of ``next()`` before the iteration loop. Deduplicates
        filenames while preserving order (a multi-sample file may appear many
        times in the thread's file_map entries).
        """
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        paths = []
        for _, filename, _ in thread_entries:
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)

        if not paths:
            return

        mode = "s3dlio-direct://" if self._use_direct else "buffered"
        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"prefetching {len(paths)} local files [{mode}]"
        )

        if self._use_direct:
            cache = self._prefetch_direct(paths)
        else:
            cache = self._prefetch_buffered(paths)

        self._total_bytes_read += sum(cache.values())
        self._total_objects_read += len(cache)
        self._local_cache = cache

    def _localfs_ensure_cached(self, filename: str) -> None:
        """Fetch a single file on demand if not already in the cache."""
        if filename not in self._local_cache:
            if self._use_direct:
                self._local_cache.update(self._prefetch_direct([filename]))
            else:
                self._local_cache[filename] = self._read_local_bytes(filename)

    def finalize_local_bytes(self) -> None:
        """
        Update ``args.record_length`` from actual bytes read this epoch.

        Mirrors ``_S3IterableMixin.finalize_s3_bytes()``. Call from subclass
        ``finalize()`` before resetting epoch state.  Resets epoch counters.
        """
        if self._total_objects_read > 0 and self._total_bytes_read > 0:
            measured = self._total_bytes_read // self._total_objects_read
            self._args.record_length = measured
            self.logger.debug(
                f"{utcnow()} {self.__class__.__name__} epoch done: "
                f"actual {self._total_bytes_read / 1024**3:.3f} GiB read, "
                f"{self._total_objects_read} files, "
                f"{measured:,} bytes/file"
            )
        self._total_bytes_read = 0
        self._total_objects_read = 0

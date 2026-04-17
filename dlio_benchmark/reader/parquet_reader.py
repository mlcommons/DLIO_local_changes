"""
Parquet reader for local and network filesystems (non-object-storage).

Reads parquet files via pyarrow directly. Each file is opened by reading its
footer (column + row-group metadata), then individual row groups are fetched on
demand as DLIO requests specific sample indices. Row groups are cached with an
LRU bound so consecutive samples from the same row group cost only one read.

This reader is the filesystem counterpart to ParquetReaderS3Iterable. Both use
identical sample-index → row-group mapping (bisect on cumulative offsets), the
same row_group_cache_size option, and the same column-selection option, so
benchmarks can switch between local and S3 storage with no config changes beyond
storage_type.

Configuration (under storage_options in the DLIO YAML):
  columns:              null  # list of column names to read (null = all)
  row_group_cache_size: 4     # max row groups held in memory per reader thread
  metadata_cache:       true  # cache parquet footer metadata across opens
  memory_map:           true  # use memory-mapped I/O
  file_cache:           true  # keep 1 ParquetFile open across close/open calls

Example YAML snippet:
  dataset:
    format: parquet
    storage_type: local
    num_samples_per_file: 1024  # must equal actual rows-per-parquet-file
    storage_options:
      columns: ["feature1", "label"]
      row_group_cache_size: 8
      metadata_cache: true
      memory_map: true
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Row-group-granular Parquet reader for local/network filesystems.

    Opens parquet files with pyarrow natively (no object-storage adapters needed).
    Row groups are cached in an LRU-bounded dict; only compressed byte counts are
    stored for the image_size telemetry metric — the actual row data is discarded
    since DLIO's FormatReader.next() always yields self._args.resized_image.

    DLIO's FormatReader protocol:
      open(filename)            → returns (ParquetFile, cumulative_offsets)
      get_sample(filename, idx) → bisect-locates the row group, fetches if not
                                  cached, updates dlp metrics with byte count
      close(filename)           → evicts row-group cache entries for that file
      next() / read_index()     → delegate to FormatReader base class
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        opts = getattr(self._args, "storage_options", {}) or {}

        # Configuration flags
        self._use_metadata_cache = opts.get("metadata_cache", True)
        self._use_memory_map = opts.get("memory_map", True)
        self._use_file_cache = opts.get("file_cache", True)

        # Metadata cache: filename -> (FileMetaData, cumulative_offsets)
        # Caches parquet footer metadata to avoid re-reading it on every open
        self._metadata_cache: dict = {}

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Row-group cache: (filename, rg_idx) → (pyarrow.Table, compressed_bytes)
        self._rg_cache_size = int(opts.get("row_group_cache_size", 4))
        self._rg_cache: dict = {}
        self._rg_lru: list = []  # insertion-order LRU key list

        # File cache: keeps at most 1 ParquetFile open across close/open cycles
        # Stored as (filename, (pf, offsets)) or None
        self._file_cache = None

        self.logger.info(
            f"{utcnow()} ParquetReader thread={thread_index} epoch={epoch} "
            f"columns={self._columns} rg_cache_size={self._rg_cache_size} "
            f"metadata_cache={self._use_metadata_cache} memory_map={self._use_memory_map} "
            f"file_cache={self._use_file_cache}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _evict_lru(self):
        """Evict the least-recently-used row group from the cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    def _evict_rg_for_file(self, filename):
        """Drop all row-group cache entries belonging to ``filename``."""
        keys_to_remove = [k for k in self._rg_cache if k[0] == filename]
        for k in keys_to_remove:
            self._rg_cache.pop(k, None)
            if k in self._rg_lru:
                self._rg_lru.remove(k)

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Open a parquet file and read its footer metadata.

        Returns (ParquetFile, cumulative_offsets) stored in open_file_map[filename].
        cumulative_offsets[i] is the first row index of row group i;
        cumulative_offsets[-1] is the total row count.
        
        With metadata_cache=True, caches parquet metadata (footer) to avoid re-reading.
        With memory_map=True, uses memory-mapped I/O for faster access.
        With file_cache=True, returns a cached ParquetFile handle if the same
        file was the last one closed, avoiding any re-open work.
        """
        import pyarrow.parquet as pq

        # File cache hit: same file as the last one we kept open
        if self._use_file_cache and self._file_cache is not None and self._file_cache[0] == filename:
            return self._file_cache[1]

        # File cache miss with a different file cached: evict it now
        if self._use_file_cache and self._file_cache is not None:
            old_filename = self._file_cache[0]
            self._evict_rg_for_file(old_filename)
            self._file_cache = None

        cached_meta = None
        cached_offsets = None

        # Check if metadata is cached
        if self._use_metadata_cache:
            cached = self._metadata_cache.get(filename)
            if cached is not None:
                cached_meta, cached_offsets = cached

        # Open the file - pass cached metadata to skip footer read if available
        pf = pq.ParquetFile(
            filename, 
            memory_map=self._use_memory_map,
            metadata=cached_meta
        )

        # Use cached offsets or compute them
        if cached_offsets is not None:
            offsets = cached_offsets
        else:
            # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
            meta = pf.metadata
            offsets = [0]
            for i in range(meta.num_row_groups):
                offsets.append(offsets[-1] + meta.row_group(i).num_rows)
            
            # Cache the metadata and offsets
            if self._use_metadata_cache:
                self._metadata_cache[filename] = (meta, offsets)

        handle = (pf, offsets)

        # Populate the 1-slot file cache
        if self._use_file_cache:
            self._file_cache = (filename, handle)

        return handle

    @dlp.log
    def close(self, filename):
        """
        Close ``filename`` and evict its row-group cache entries.

        With ``file_cache`` enabled, the most recently used file is kept open;
        the actual close/eviction is deferred until a different file is opened
        (handled in :meth:`open`) or until :meth:`finalize` runs.
        """
        if self._use_file_cache and self._file_cache is not None and self._file_cache[0] == filename:
            # Keep this file open in the 1-slot file cache
            return

        self._evict_rg_for_file(filename)
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the row group containing sample_index and update I/O metrics.

        Uses bisect to locate the row group in O(log N), fetches from disk if
        not already cached. Reports compressed row-group bytes to the profiler.
        Actual row data is discarded — DLIO uses self._args.resized_image.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: offsets[rg_idx] <= sample_index < offsets[rg_idx+1]
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            # Read row group from disk — this is the measured I/O
            pf.read_row_group(rg_idx, columns=self._columns)

            rg_meta = pf.metadata.row_group(rg_idx)
            compressed_bytes = sum(
                rg_meta.column(c).total_compressed_size
                for c in range(rg_meta.num_columns)
            )

            while len(self._rg_cache) >= self._rg_cache_size:
                self._evict_lru()

            self._rg_cache[cache_key] = compressed_bytes
            self._rg_lru.append(cache_key)
        else:
            # Move to end (most recently used)
            try:
                self._rg_lru.remove(cache_key)
            except ValueError:
                pass
            self._rg_lru.append(cache_key)

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self._rg_cache.clear()
        self._rg_lru.clear()
        self._file_cache = None
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

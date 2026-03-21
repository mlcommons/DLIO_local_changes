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

Example YAML snippet:
  dataset:
    format: parquet
    storage_type: local
    num_samples_per_file: 1024  # must equal actual rows-per-parquet-file
    storage_options:
      columns: ["feature1", "label"]
      row_group_cache_size: 8
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

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Row-group cache: (filename, rg_idx) → (pyarrow.Table, compressed_bytes)
        self._rg_cache_size = int(opts.get("row_group_cache_size", 4))
        self._rg_cache: dict = {}
        self._rg_lru: list = []  # insertion-order LRU key list

        self.logger.info(
            f"{utcnow()} ParquetReader thread={thread_index} epoch={epoch} "
            f"columns={self._columns} rg_cache_size={self._rg_cache_size}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _evict_lru(self):
        """Evict the least-recently-used row group from the cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Open a parquet file and read its footer metadata.

        Returns (ParquetFile, cumulative_offsets) stored in open_file_map[filename].
        cumulative_offsets[i] is the first row index of row group i;
        cumulative_offsets[-1] is the total row count.
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename)
        meta = pf.metadata

        # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReader.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )
        return (pf, offsets)

    @dlp.log
    def close(self, filename):
        """Evict cached row groups for this file to free memory."""
        keys_to_remove = [k for k in self._rg_cache if k[0] == filename]
        for k in keys_to_remove:
            self._rg_cache.pop(k, None)
            if k in self._rg_lru:
                self._rg_lru.remove(k)
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
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

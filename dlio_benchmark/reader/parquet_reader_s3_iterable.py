"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
"""
Parquet reader for S3-compatible object storage using HTTP byte-range GET requests.

Each parquet file may contain many rows (samples) and multiple columns (features).
Reads are row-group-granular: pyarrow.parquet.ParquetFile opens the file by reading
only the footer (a small range request for column and row-group metadata). Individual
row groups are then fetched on demand via server-side Range requests, avoiding full
file downloads.

Supported storage libraries
  s3dlio           — uses s3dlio.get_range(uri, offset, length) and s3dlio.stat(uri)
  s3torchconnector — same as s3dlio (uses s3dlio as the underlying engine)
  minio            — uses minio.Minio.get_object(bucket, key, offset=, length=)

Configuration (under storage_options in the DLIO YAML):
  storage_library:      s3dlio      # or s3torchconnector / minio
  endpoint_url:         http://...  # S3 endpoint; also settable via AWS_ENDPOINT_URL_S3
  columns:              null        # list of column names to read (null = all)
  row_group_cache_size: 4           # max row groups to hold in memory per reader

Example YAML snippet:
  dataset:
    format: parquet
    storage_type: s3
    storage_root: my-bucket
    num_samples_per_file: 1024  # must equal actual rows-per-parquet-file
    storage_options:
      storage_library: s3dlio
      endpoint_url: http://127.0.0.1:9000
      columns: ["feature1", "label"]
      row_group_cache_size: 8
"""
import bisect
import os

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


# ── Seekable file-like adapters ───────────────────────────────────────────────


class _S3RangeFile:
    """
    Seekable, readable file-like object backed by s3dlio byte-range GETs.

    Used for both s3dlio and s3torchconnector (s3dlio is the underlying engine
    in both cases). pyarrow.parquet.ParquetFile passes this to its C++ reader
    which calls seek/tell/read as needed when scanning column chunks.
    """

    def __init__(self, uri: str):
        self._uri = uri
        self._pos = 0
        self._size = None  # fetched lazily on first seek-from-end or full-read

    def _ensure_size(self):
        if self._size is None:
            import s3dlio
            self._size = s3dlio.stat(self._uri)["size"]

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._ensure_size()
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        self._ensure_size()
        remaining = self._size - self._pos
        if remaining <= 0:
            return b""
        if n < 0 or n > remaining:
            n = remaining
        import s3dlio
        data = s3dlio.get_range(self._uri, self._pos, n)
        self._pos += n
        return bytes(data)

    def readall(self) -> bytes:
        return self.read(-1)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self):
        pass


class _MinioRangeFile:
    """
    Seekable, readable file-like object backed by minio byte-range GETs.

    Uses minio.Minio.get_object(bucket, key, offset=offset, length=length)
    for each read() call, matching the s3dlio interface semantics.
    """

    def __init__(self, bucket: str, key: str, client):
        self._bucket = bucket
        self._key = key
        self._client = client
        self._pos = 0
        self._size = None

    def _ensure_size(self):
        if self._size is None:
            self._size = self._client.stat_object(self._bucket, self._key).size

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._ensure_size()
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        self._ensure_size()
        remaining = self._size - self._pos
        if remaining <= 0:
            return b""
        if n < 0 or n > remaining:
            n = remaining
        resp = self._client.get_object(
            self._bucket, self._key, offset=self._pos, length=n
        )
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        self._pos += len(data)
        return data

    def readall(self) -> bytes:
        return self.read(-1)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def close(self):
        pass


# ── Main reader ───────────────────────────────────────────────────────────────


class ParquetReaderS3Iterable(FormatReader):
    """
    Row-group-granular Parquet reader for S3-compatible object storage.

    Opens parquet files by reading only the footer (column / row-group metadata)
    via a small range request, then fetches individual row groups on demand as
    DLIO requests specific sample indices.  Row groups are cached (LRU-bounded)
    so that consecutive samples from the same row group incur only one network
    round-trip.

    DLIO's FormatReader protocol:
      open(filename)               → returns (ParquetFile, cumulative_offsets)
                                     stored in self.open_file_map[filename]
      get_sample(filename, idx)    → looks up the right row group, fetches if
                                     not cached, updates dlp metrics
      close(filename)              → evicts row-group cache entries for that file
      next() / read_index()        → delegate to FormatReader base class

    The cumulative_offsets list has len(num_row_groups + 1) entries; entry i
    is the first global row index of row group i.  Binary search maps a sample
    index to (rg_idx, within-row-group offset) in O(log num_row_groups).
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._opts = opts
        self._epoch = epoch

        # Optional column selection (list[str] or None = all columns)
        self._columns = opts.get("columns") or None

        # Row-group cache: (filename, rg_idx) → (pyarrow.Table, nbytes)
        self._rg_cache_size = int(opts.get("row_group_cache_size", 4))
        self._rg_cache: dict = {}
        self._rg_lru: list = []  # insertion-order LRU key list

        # Configure s3dlio endpoint at construction time
        if self._storage_library in ("s3dlio", "s3torchconnector"):
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        # Minio client created lazily once, reused across files
        self._minio_client = None

        self.logger.info(
            f"{utcnow()} ParquetReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch} "
            f"columns={self._columns} rg_cache_size={self._rg_cache_size}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _uri_for_filename(self, filename: str) -> str:
        """Return a full s3:// URI for a DLIO filename (relative or absolute)."""
        if "://" in filename:
            return filename
        root = self._args.storage_root.rstrip("/")
        return f"s3://{root}/{filename.lstrip('/')}"

    def _get_minio_client(self):
        if self._minio_client is None:
            from minio import Minio

            opts = self._opts
            endpoint = opts.get("endpoint_url", "")
            if endpoint.startswith("https://"):
                host, secure = endpoint[8:], True
            elif endpoint.startswith("http://"):
                host, secure = endpoint[7:], False
            else:
                host, secure = endpoint, False
            self._minio_client = Minio(
                host,
                access_key=opts.get("access_key_id"),
                secret_key=opts.get("secret_access_key"),
                secure=secure,
                region=opts.get("region", "us-east-1"),
            )
        return self._minio_client

    def _make_range_file(self, filename: str):
        """Create a seekable file-like object for the given filename."""
        uri = self._uri_for_filename(filename)
        lib = self._storage_library
        if lib in ("s3dlio", "s3torchconnector"):
            return _S3RangeFile(uri)
        elif lib == "minio":
            from urllib.parse import urlparse

            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return _MinioRangeFile(bucket, key, self._get_minio_client())
        else:
            raise ValueError(
                f"ParquetReaderS3Iterable: unknown storage_library {lib!r}; "
                "supported: s3dlio, s3torchconnector, minio"
            )

    def _evict_lru(self):
        """Evict the least-recently-used row group from the cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Open a parquet file by reading its footer via a small range request.

        Returns a tuple (ParquetFile, cumulative_offsets) stored in
        open_file_map[filename].  cumulative_offsets[i] is the first row index
        of row group i; cumulative_offsets[-1] is the total row count.
        """
        import pyarrow.parquet as pq

        rf = self._make_range_file(filename)
        pf = pq.ParquetFile(rf)
        meta = pf.metadata

        # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReaderS3Iterable.open {filename} "
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

        Uses bisect to locate the row group in O(log N), then fetches the row
        group from object storage if not already in the row-group cache.
        Actual row data is read but the DLIO pipeline uses a pre-allocated
        random tensor (self._args.resized_image) for the training simulation;
        we report the compressed row-group bytes to the profiler.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: find rg_idx such that offsets[rg_idx] <= sample_index
        # < offsets[rg_idx + 1].  bisect_right on offsets gives insertion point
        # for sample_index+1, so rg_idx = that - 1, clamped to valid range.
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            # Fetch this row group — triggers range GETs for column chunks
            table = pf.read_row_group(rg_idx, columns=self._columns)

            # Report the uncompressed bytes actually transferred/processed
            rg_meta = pf.metadata.row_group(rg_idx)
            compressed_bytes = sum(
                rg_meta.column(c).total_compressed_size
                for c in range(rg_meta.num_columns)
            )

            # LRU eviction when cache is full
            while len(self._rg_cache) >= self._rg_cache_size:
                self._evict_lru()

            self._rg_cache[cache_key] = (table, compressed_bytes)
            self._rg_lru.append(cache_key)
        else:
            # Move to end (most recently used)
            try:
                self._rg_lru.remove(cache_key)
            except ValueError:
                pass
            self._rg_lru.append(cache_key)

        _, compressed_bytes = self._rg_cache[cache_key]
        dlp.update(image_size=compressed_bytes)

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

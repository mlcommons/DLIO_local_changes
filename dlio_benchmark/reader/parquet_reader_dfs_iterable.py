"""
   Copyright (c) 2026, Gluesys Co., Ltd.
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
Row-group-granular Parquet reader for storage_type: daos (libdfs, no dfuse).

Same measurement as ParquetReaderFileIterable and ParquetReaderS3Iterable:
only the compressed column-chunk bytes of each row group are read (one
range read per row group), nothing is decoded, and the compressed byte
count is reported to telemetry.  The byte ranges come from the same footer
arithmetic as ParquetReaderFileIterable.

The footer is parsed by ``pyarrow.parquet.ParquetFile`` over a seekable
libdfs file object (``DfsClient.open_read``), which is closed as soon as the
footer is parsed; only the parsed metadata is kept.  No container path is
ever opened as a local file.

Exposes ``open_footer_only()`` / ``submit_rg_prefetch()`` and
``_prefetch_executor`` so that TorchIterableDataset drives it through the
same sliding-window path as the file and S3 readers.

Configuration (under storage.storage_options):
  prefetch_workers: 64   # range-read threads per worker process (max 64)
  columns:          null # list of column names (null = all)
  prefetch_window:  64   # sliding-window depth (TorchIterableDataset)
"""
import bisect
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import ReadType as _ReadType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader.parquet_reader_file_iterable import ParquetReaderFileIterable
from dlio_benchmark.storage.dfs_client import get_client
from dlio_benchmark.storage.dfs_precheck import daos_connection_options
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)

_MAX_PREFETCH_CONCURRENCY = 64


class ParquetReaderDfsIterable(FormatReader):
    """Row-group range reads from DAOS (storage_type: daos).

    Cache format: ``_pf_cache[filename] = (pf, None, offsets)``, the same
    as ParquetReaderFileIterable.  ``pf`` is used only for its parsed
    metadata (``pf.metadata``, ``pf.schema_arrow``); its source is closed.
    """

    _rg_byte_range = ParquetReaderFileIterable._rg_byte_range

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        opts = getattr(self._args, "storage_options", {}) or {}
        self._columns = opts.get("columns") or None
        self._pf_cache: dict = {}
        self._rg_cache: dict = {}
        self._prefetch_futures: dict = {}

        pool, container, sys_name = daos_connection_options(self._args)
        self._dfs_client = get_client(pool, container, sys_name)

        max_w = max(1, min(_MAX_PREFETCH_CONCURRENCY, int(opts.get("prefetch_workers", 64))))
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=max_w, thread_name_prefix="rg-dfs-prefetch"
        )

        self.logger.info(
            f"{utcnow()} ParquetReaderDfsIterable "
            f"thread={thread_index} epoch={epoch} "
            f"prefetch_workers={max_w} columns={self._columns}"
        )

    # -- I/O helpers ------------------------------------------------------------

    def _fetch_rg_dfs(self, filename: str, rg_start: int, rg_len: int, compressed_bytes: int) -> int:
        """Range-read one row group's compressed bytes and discard them."""
        if rg_len > 0:
            buf = np.empty(rg_len, dtype=np.uint8)
            n = self._dfs_client.readinto(filename, buf, rg_start)
            del buf
            if n != rg_len:
                raise OSError(
                    f"short row-group read from {filename}: {n} of {rg_len} bytes "
                    f"at offset {rg_start}")
        return compressed_bytes

    def _open_footer(self, filename):
        import pyarrow.parquet as pq

        with self._dfs_client.open_read(filename) as source:
            pf = pq.ParquetFile(source)
        meta = pf.metadata
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)
        return pf, None, offsets

    def _submit(self, filename, pf, rg_idx):
        rg_start_b, rg_len, all_comp = self._rg_byte_range(pf, rg_idx)
        return self._prefetch_executor.submit(
            self._fetch_rg_dfs, filename, rg_start_b, rg_len, all_comp
        )

    # -- Sliding-window helpers (same API as ParquetReaderFileIterable) --------

    def open_footer_only(self, filename):
        """Read the footer; return ``(pf, None, offsets)`` without prefetching."""
        if filename in self._pf_cache:
            return self._pf_cache[filename]
        self._pf_cache[filename] = self._open_footer(filename)
        return self._pf_cache[filename]

    def submit_rg_prefetch(self, filename, rg_idx):
        """Submit one row-group range read; idempotent per ``(filename, rg_idx)``."""
        key = (filename, rg_idx)
        if key in self._prefetch_futures:
            return self._prefetch_futures[key]
        pf, _, _ = self._pf_cache[filename]
        fut = self._submit(filename, pf, rg_idx)
        self._prefetch_futures[key] = fut
        return fut

    # -- FormatReader interface -------------------------------------------------

    @dlp.log
    def open(self, filename):
        """Footer at most once per epoch; submits every row group of the file."""
        if filename in self._pf_cache:
            return self._pf_cache[filename]
        pf, _, offsets = self._open_footer(filename)
        self.logger.debug(
            f"{utcnow()} ParquetReaderDfsIterable.open {filename} "
            f"row_groups={pf.metadata.num_row_groups} total_rows={offsets[-1]}"
        )
        for rg_i in range(pf.metadata.num_row_groups):
            key = (filename, rg_i)
            if key not in self._prefetch_futures:
                self._prefetch_futures[key] = self._submit(filename, pf, rg_i)
        self._pf_cache[filename] = (pf, None, offsets)
        return self._pf_cache[filename]

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """Wait for the row group holding ``sample_index``; record its compressed bytes."""
        pf, _, offsets = self.open_file_map[filename]

        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            if cache_key in self._prefetch_futures:
                compressed_bytes = self._prefetch_futures.pop(cache_key).result()
            else:
                rg_start_b, rg_len, all_comp = self._rg_byte_range(pf, rg_idx)
                compressed_bytes = self._fetch_rg_dfs(filename, rg_start_b, rg_len, all_comp)
            self._rg_cache[cache_key] = compressed_bytes

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, sample_index = self.global_index_map[image_idx]
        if (
            filename not in self.open_file_map
            or self.open_file_map[filename] is None
        ):
            self.open_file_map[filename] = self.open(filename)
        self.get_sample(filename, sample_index)
        if self._args.read_type is _ReadType.ON_DEMAND:
            self.open_file_map[filename] = None
        return self._args.resized_image

    @dlp.log
    def finalize(self):
        """Cancel outstanding reads and flush caches."""
        for fut in self._prefetch_futures.values():
            fut.cancel()
        self._prefetch_futures.clear()
        self._pf_cache.clear()
        self._rg_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

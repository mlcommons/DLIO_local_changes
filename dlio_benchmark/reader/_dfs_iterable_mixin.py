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
_DfsIterableMixin -- in-order read-ahead for ``storage_type: daos``.

Same behaviour as ``_LocalFSIterableMixin`` and ``_S3IterableMixin``
towards the data loaders:

1. ``_dfs_stream_next()`` walks this thread's ``file_map`` entries in
   order.  While one file is consumed, up to
   ``storage_options.prefetch_window`` (default 4) later files are being
   read or wait to be consumed.  Reads run on one thread pool per reader
   (one thread per read-ahead file, at most 64), created on first use and
   reused until ``finalize()`` shuts it down; the next read creates a new
   one.  The byte counters cover files whose samples were consumed; files
   read ahead and dropped by an early close are not counted.  A read error
   surfaces when the failing file reaches the head of the queue, after the
   files before it were consumed.
2. Every file is read in full through libdfs into a preallocated buffer
   (``dfs_read`` fills it directly; no Python-side copy) and only its byte
   count is kept.  Nothing is decoded.
3. ``next()`` yields the pre-built ``args.resized_image`` once per batch;
   ``get_sample()`` reports the byte count as ``image_size``.

The DFS connection belongs to the process (``dfs_client.get_client``).
In DataLoader workers it is opened by ``worker_init`` and closed when the
worker process exits.  The reader keeps no state across epochs other than
the byte counters reset by ``finalize_dfs_bytes()`` and the idle read
pool: each ``next()`` call reads the file list currently installed in
``file_map``.  Closing a read-ahead generator waits for its running reads;
reads not started yet are cancelled.

Subclass from ``FormatReader`` and this mixin only; never from a local
reader, whose open() path would treat container paths as local files.
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dlio_benchmark.storage.dfs_client import get_client
from dlio_benchmark.storage.dfs_precheck import daos_connection_options
from dlio_benchmark.utils.utility import utcnow

_MAX_PREFETCH_CONCURRENCY = 64
_DEFAULT_PREFETCH_WINDOW = 4


class _DfsIterableMixin:
    """Mixin providing DFS read-ahead for iterable readers.

    Call ``_dfs_init()`` from the subclass ``__init__`` after
    ``FormatReader.__init__``.
    """

    def _dfs_init(self) -> None:
        """
        Sets:
          - ``self._dfs_cache``          (dict: path -> int byte count)
          - ``self._dfs_client``         (the process's DfsClient)
          - ``self._dfs_read_pool``      (None until the first read-ahead)
          - ``self._total_bytes_read``   (int, epoch accumulator)
          - ``self._total_objects_read`` (int, epoch accumulator)
        """
        self._dfs_cache: dict = {}
        self._total_bytes_read: int = 0
        self._total_objects_read: int = 0
        pool, container, sys_name = daos_connection_options(self._args)
        self._dfs_client = get_client(pool, container, sys_name)
        self._dfs_read_pool = None

    # -- reads -----------------------------------------------------------------

    def _dfs_read_one(self, path: str) -> int:
        """Read ``path`` in full and return its byte count.  No decode."""
        with self._dfs_client.open_read(path) as f:
            size = f.size()
            if size == 0:
                return 0
            buf = np.empty(size, dtype=np.uint8)
            n = f.readinto(buf)
            del buf
            return n

    def _dfs_account(self, nbytes: int) -> None:
        self._total_bytes_read += nbytes
        self._total_objects_read += 1

    def _dfs_pool(self) -> ThreadPoolExecutor:
        """The reader's read pool, created on first use."""
        if self._dfs_read_pool is None:
            workers = min(self._dfs_window_size(), _MAX_PREFETCH_CONCURRENCY)
            self._dfs_read_pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="dfs_read")
        return self._dfs_read_pool

    def _dfs_shutdown(self) -> None:
        """Shut the read pool down; the next read-ahead creates a new one.

        Queued reads are cancelled and the call does not wait for running
        ones: a read-ahead generator that is closed later waits for its own
        running reads.  Idempotent.
        """
        pool, self._dfs_read_pool = self._dfs_read_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _dfs_readahead(self, paths: list, window: int):
        """Yield ``(path, byte_count)`` for ``paths`` in order.

        While the caller handles the yielded file, up to ``window`` later
        files are being read or have been read and wait to be yielded.  The
        byte counters cover yielded files only; files read ahead and dropped
        by an early close are not counted.  A read error is raised when the
        failing file reaches the head of the queue, after every file before
        it has been yielded.  On close or error, queued reads are cancelled
        and running ones are waited for, so no read is in flight afterwards.
        Each future is drained on its own: a future cancelled by
        ``_dfs_shutdown()`` never reaches a state that ``wait()`` counts as
        done.
        """
        pool = self._dfs_pool()
        remaining = iter(paths)
        pending = deque()

        def refill():
            while len(pending) < window:
                path = next(remaining, None)
                if path is None:
                    return
                pending.append((path, pool.submit(self._dfs_read_one, path)))

        try:
            refill()
            while pending:
                path, future = pending[0]
                nbytes = future.result()
                pending.popleft()
                self._dfs_account(nbytes)
                refill()
                yield path, nbytes
        finally:
            for _, f in pending:
                if not f.cancel():
                    # Running or finished: wait for it; its outcome is dropped
                    # so that it cannot replace the exception being raised.
                    f.exception()

    def _dfs_thread_paths(self) -> list:
        """Unique file paths of this thread's file_map, in order."""
        thread_entries = self.file_map.get(self.thread_index, []) or []
        seen = set()
        paths = []
        for _, filename, _ in thread_entries:
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)
        return paths

    # -- iteration -------------------------------------------------------------

    def _dfs_window_size(self) -> int:
        opts = getattr(self._args, "storage_options", {}) or {}
        return max(1, int(opts.get("prefetch_window", _DEFAULT_PREFETCH_WINDOW)))

    def _dfs_stream_next(self):
        """In-order read-ahead of this thread's files; one yield per batch."""
        args = self._args
        batch_size = self.batch_size
        num_spf = args.num_samples_per_file
        opts = getattr(args, "storage_options", {}) or {}
        window = self._dfs_window_size()
        dummy = args.resized_image

        paths = self._dfs_thread_paths()
        if not paths:
            return

        if str(opts.get("stagger_workers", "true")).lower() not in ("false", "0", "no"):
            ct_raw = getattr(args, "computation_time", 0.0)
            ct = ct_raw.get("mean", 0.0) if isinstance(ct_raw, dict) else float(ct_raw or 0.0)
            delay = self.thread_index * ct
            if delay > 0:
                self.logger.debug(
                    f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
                    f"stagger delay={delay:.4f}s ({self.thread_index} x {ct:.4f}s)"
                )
                time.sleep(delay)

        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"streaming {len(paths)} files, read-ahead {window} [daos]"
        )

        sample_buf = 0
        files = self._dfs_readahead(paths, window)
        try:
            for path, nbytes in files:
                self._dfs_cache = {path: nbytes}
                for s in range(num_spf):
                    self.get_sample(path, s)
                    sample_buf += 1
                    if sample_buf >= batch_size:
                        yield dummy
                        sample_buf -= batch_size
                self._dfs_cache = {}
        finally:
            files.close()
            self._dfs_cache = {}
        # Drop-last: a remaining partial batch is discarded.

    def _dfs_ensure_cached(self, filename: str) -> None:
        """Read one file on demand (map-style access); always re-reads."""
        nbytes = self._dfs_read_one(filename)
        self._dfs_account(nbytes)
        self._dfs_cache[filename] = nbytes

    def finalize_dfs_bytes(self) -> None:
        """Set ``args.record_length`` from this epoch's bytes; reset counters."""
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

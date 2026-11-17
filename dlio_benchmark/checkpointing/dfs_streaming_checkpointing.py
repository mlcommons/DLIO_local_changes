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
Streaming checkpoint backend for storage_type: daos (libdfs, no dfuse).

Writes and reads the same data, in the same pattern, as mlpstorage's
``StreamingCheckpointing`` (used for ``file`` and ``direct_fs``), through
libdfs instead of a POSIX file.  ``uri`` is a path inside the DAOS container
(``checkpoint_folder`` joined with the checkpoint file name); it is resolved
from the container root.

save: one dgen-py ``Generator`` per file (``dedup_ratio=1.0``,
      ``compress_ratio=1.0``, threads = CPUs / ranks per node), so every
      chunk is fresh, incompressible data.  A producer thread fills
      ``num_buffers`` buffers of ``chunk_size`` bytes in turn; the calling
      thread writes each filled buffer at the next offset through one write
      handle and returns it to the producer once ``write`` has returned.
      The file is created (or truncated) once.  DFS writes are synchronous
      and there is no client-side page cache, so no separate flush is
      issued.  Nothing runs in a subprocess: a forked child must not reuse
      the parent's DAOS client state.
load: the file is split into ``num_parallel_readers`` contiguous blocks of
      ``ceil(total / num_parallel_readers)`` bytes; one thread per non-empty
      block opens its own read handle and reads the block in
      ``read_chunk_size`` pieces (default 4 x ``chunk_size``) into one
      reusable buffer.  A file shorter than ``total_bytes`` raises
      ``OSError``.
"""

import os
import queue
import threading

import numpy as np

from dlio_benchmark.storage.dfs_client import normalize_path

try:
    import dgen_py as _dgen_py
except ImportError:
    _dgen_py = None

_CHUNK_SIZE = 32 * 1024 * 1024


def _ranks_per_node():
    """MPI ranks sharing this node (>= 1); 1 when MPI is not initialized."""
    try:
        from dlio_benchmark.utils.utility import DLIOMPI
        return max(1, int(DLIOMPI.get_instance().ranks_per_node()))
    except Exception:
        return 1


class DfsStreamingCheckpointing:
    def __init__(self, chunk_size=_CHUNK_SIZE, num_buffers=4, num_parallel_readers=4,
                 read_chunk_size=None, client=None, **_kwargs):
        self.chunk_size = max(1024 * 1024, int(chunk_size))
        self.num_buffers = max(1, int(num_buffers))
        self.num_parallel_readers = max(1, int(num_parallel_readers))
        self.read_chunk_size = (int(read_chunk_size) if read_chunk_size
                                else 4 * self.chunk_size)
        self._client = client
        self._buffers = None

    @property
    def client(self):
        if self._client is None:
            from dlio_benchmark.storage.dfs_client import get_client
            from dlio_benchmark.storage.dfs_precheck import daos_connection_options
            from dlio_benchmark.utils.config import ConfigArguments
            return get_client(*daos_connection_options(ConfigArguments.get_instance()))
        return self._client

    def _open_write(self, client, path):
        try:
            return client.open_write(path)
        except FileNotFoundError:
            parent = path.rsplit("/", 1)[0] or "/"
            client.mkdir(parent, parents=True, exist_ok=True)
            return client.open_write(path)

    # -- save ------------------------------------------------------------------

    def _generator(self, total):
        if _dgen_py is None:
            raise ImportError(
                "dgen-py is required for storage_type daos checkpoint data. "
                "Install with: pip install dgen-py")
        max_threads = max(1, (os.cpu_count() or 4) // _ranks_per_node())
        return _dgen_py.Generator(
            size=total,
            chunk_size=self.chunk_size,
            dedup_ratio=1.0,
            compress_ratio=1.0,
            numa_mode="auto",
            max_threads=max_threads,
        )

    def _write_buffers(self):
        if self._buffers is None:
            self._buffers = [np.empty(self.chunk_size, dtype=np.uint8)
                             for _ in range(self.num_buffers)]
        return self._buffers

    def save(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        total = int(total_size_bytes)
        path = normalize_path(uri)
        client = self.client
        generator = self._generator(total)
        buffers = self._write_buffers()

        free = queue.Queue()
        ready = queue.Queue()
        stop = threading.Event()
        producer_errors = []
        for i in range(len(buffers)):
            free.put(i)

        def produce():
            try:
                offset = 0
                while offset < total:
                    i = free.get()
                    if i is None or stop.is_set():
                        return
                    n = min(self.chunk_size, total - offset)
                    filled = generator.fill_chunk(memoryview(buffers[i])[:n])
                    if filled != n:
                        raise RuntimeError(
                            f"dgen-py filled {filled} of {n} bytes at offset {offset} "
                            f"of checkpoint '{path}'")
                    ready.put((i, n))
                    offset += n
            except BaseException as exc:  # noqa: BLE001 - re-raised by save()
                producer_errors.append(exc)
                ready.put(None)

        writer = self._open_write(client, path)
        producer = threading.Thread(target=produce, name="dfs_ckpt_fill", daemon=True)
        error = None
        written = 0
        try:
            producer.start()
            while written < total:
                item = ready.get()
                if item is None:
                    break
                i, n = item
                writer.write(memoryview(buffers[i])[:n])
                written += n
                free.put(i)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error = exc
        finally:
            stop.set()
            free.put(None)
            if producer.ident is not None:
                producer.join()
            # Do not hold num_buffers x chunk_size between saves.
            self._buffers = None
            try:
                writer.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if error is None:
                    error = exc
        if error is None and producer_errors:
            error = producer_errors[0]
        if error is None and written != total:
            error = RuntimeError(
                f"checkpoint '{path}': wrote {written} of {total} bytes")
        if error is not None:
            raise error

    # -- load ------------------------------------------------------------------

    @staticmethod
    def _read_block(client, path, start, end, piece):
        buf = np.empty(piece, dtype=np.uint8)
        view = memoryview(buf)
        with client.open_read(path) as f:
            f.seek(start)
            pos = start
            while pos < end:
                n = min(piece, end - pos)
                got = f.readinto(view[:n])
                if not got:
                    raise OSError(
                        f"Checkpoint file ended early while reading '{path}'. "
                        f"Remaining bytes: {end - pos} of block [{start}, {end})"
                    )
                pos += got

    def load(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        total = int(total_size_bytes)
        path = normalize_path(uri)
        client = self.client
        block = -(-total // self.num_parallel_readers)
        blocks = [(start, min(start + block, total)) for start in range(0, total, block)]
        piece = min(self.read_chunk_size, block)
        if len(blocks) == 1:
            self._read_block(client, path, blocks[0][0], blocks[0][1], piece)
            return

        errors = [None] * len(blocks)

        def run(index, start, end):
            try:
                self._read_block(client, path, start, end, piece)
            except BaseException as exc:  # noqa: BLE001 - re-raised by load()
                errors[index] = exc

        threads = [threading.Thread(target=run, args=(index, start, end),
                                    name=f"dfs_ckpt_read_{index}", daemon=True)
                   for index, (start, end) in enumerate(blocks)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for exc in errors:
            if exc is not None:
                raise exc

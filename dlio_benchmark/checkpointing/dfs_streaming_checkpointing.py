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

Same ``save(uri, total_bytes)`` / ``load(uri, total_bytes)`` contract as
SimpleStreamingCheckpointing.  ``uri`` is a path inside the DAOS container
(``checkpoint_folder`` joined with the checkpoint file name); it is resolved
from the container root.

save: the file is created (or truncated) once and written in ``chunk_size``
      pieces at consecutive offsets.  DFS writes are synchronous and there is
      no client-side page cache, so no separate flush is issued.
load: ``chunk_size`` range reads until ``total_bytes`` have been read; a file
      shorter than ``total_bytes`` raises ``OSError``.
"""

import numpy as np

from dlio_benchmark.storage.dfs_client import normalize_path

_CHUNK_SIZE = 32 * 1024 * 1024


class DfsStreamingCheckpointing:
    def __init__(self, chunk_size=_CHUNK_SIZE, client=None, **_kwargs):
        self.chunk_size = max(1024 * 1024, int(chunk_size))
        self._client = client
        self._chunk = None

    @property
    def client(self):
        if self._client is None:
            from dlio_benchmark.storage.dfs_client import get_client
            from dlio_benchmark.storage.dfs_precheck import daos_connection_options
            from dlio_benchmark.utils.config import ConfigArguments
            return get_client(*daos_connection_options(ConfigArguments.get_instance()))
        return self._client

    def _write_chunk(self):
        if self._chunk is None:
            rng = np.random.default_rng()
            self._chunk = rng.integers(0, 256, size=self.chunk_size, dtype=np.uint8)
        return self._chunk

    def _open_write(self, client, path):
        try:
            return client.open_write(path)
        except FileNotFoundError:
            parent = path.rsplit("/", 1)[0] or "/"
            client.mkdir(parent, parents=True, exist_ok=True)
            return client.open_write(path)

    def save(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        path = normalize_path(uri)
        client = self.client
        chunk = self._write_chunk()
        stamp = chunk[:8].view(np.uint64)
        remaining = int(total_size_bytes)
        index = 0
        with self._open_write(client, path) as w:
            while remaining > 0:
                n = min(remaining, self.chunk_size)
                stamp[0] = index
                w.write(chunk[:n])
                remaining -= n
                index += 1

    def load(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        path = normalize_path(uri)
        client = self.client
        remaining = int(total_size_bytes)
        buf = np.empty(min(self.chunk_size, remaining), dtype=np.uint8)
        with client.open_read(path) as f:
            while remaining > 0:
                n = min(remaining, self.chunk_size)
                got = f.readinto(buf[:n])
                if not got:
                    raise OSError(
                        f"Checkpoint file ended early while reading '{path}'. "
                        f"Remaining bytes: {remaining}"
                    )
                remaining -= got

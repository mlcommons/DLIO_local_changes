"""
Simple fallback streaming checkpoint backend for file/direct_fs paths.

This backend is used when mlpstorage is not available. It preserves the
save/load byte-count semantics required by DLIO checkpoint tests without
introducing a runtime dependency on mlpstorage.
"""

import os

from dlio_benchmark.storage.file_storage import _makedirs_race_safe


class SimpleStreamingCheckpointing:
    def __init__(self, chunk_size=32 * 1024 * 1024, backend="file", **_kwargs):
        self.chunk_size = max(1024 * 1024, int(chunk_size))
        self.backend = backend
        self._zero_chunk = b"\x00" * self.chunk_size

    def _resolve_path(self, uri):
        if uri.startswith("direct://"):
            return uri[len("direct://"):]
        if uri.startswith("file://"):
            return uri[len("file://"):]
        return uri

    def save(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        path = self._resolve_path(uri)
        parent = os.path.dirname(path)
        if parent:
            # storage#699: multiple ranks/hosts write different checkpoint
            # files under the same step directory concurrently — same
            # multi-host makedirs race as FileStorage.create_node().
            _makedirs_race_safe(parent, exist_ok=True)

        remaining = int(total_size_bytes)
        with open(path, "wb") as f:
            while remaining > 0:
                n = min(remaining, self.chunk_size)
                f.write(self._zero_chunk[:n])
                remaining -= n
            f.flush()
            os.fsync(f.fileno())

    def load(self, uri, total_size_bytes):
        if total_size_bytes <= 0:
            return
        path = self._resolve_path(uri)
        remaining = int(total_size_bytes)
        with open(path, "rb") as f:
            while remaining > 0:
                n = min(remaining, self.chunk_size)
                data = f.read(n)
                if not data:
                    raise IOError(
                        f"Checkpoint file ended early while reading '{path}'. "
                        f"Remaining bytes: {remaining}"
                    )
                remaining -= len(data)

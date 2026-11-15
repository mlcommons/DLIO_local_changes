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
A directory-backed stand-in for ``DfsClient``, for tests without DAOS.

``get_client()`` returns a ``DirBackedDfsClient`` rooted at
``$DLIO_DAOS_TEST_BACKING_DIR/<pool>/<container>`` when that variable is set.
Because the data lives in a real directory, spawned DataLoader workers and
MPI ranks see what the parent wrote, which an in-memory fake cannot offer.

Container paths keep the ``DfsClient`` semantics: they are absolute within
the container, and results and exceptions have the same shape.
"""

import errno
import io
import os
import shutil
import threading

from dlio_benchmark.storage.dfs_client import DfsError, normalize_path


class _DirFile(io.FileIO):
    """``io.FileIO`` with the ``size()`` method of ``DfsFile``."""

    def size(self):
        return os.fstat(self.fileno()).st_size

    def readinto(self, b):
        expected = max(0, min(memoryview(b).nbytes, self.size() - self.tell()))
        n = super().readinto(b)
        if n is not None and n < expected:
            raise OSError(errno.EIO, f"short read, {n} of {expected} bytes", self.name)
        return n


class _DirWriter:
    """Sequential writer with the ``DfsWriter`` interface."""

    def __init__(self, path, local_path):
        self._path = path
        self._fh = open(local_path, "wb")

    @property
    def name(self):
        return self._path

    @property
    def closed(self):
        return self._fh.closed

    def tell(self):
        return self._fh.tell()

    def write(self, data) -> int:
        if hasattr(data, "getbuffer"):
            data = data.getbuffer()
        return self._fh.write(data)

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class DirBackedDfsClient:
    """Same public methods as ``DfsClient``; container root is ``root_dir``."""

    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self._lock = threading.Lock()
        self._connected = False
        self._closed = False

    def __repr__(self):
        return f"DirBackedDfsClient(root_dir={self.root_dir!r})"

    @property
    def connected(self):
        return self._connected

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            os.makedirs(self.root_dir, exist_ok=True)
            self._connected = True
            self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self._connected = False
            self._closed = True

    def _local(self, path):
        if not self._connected:
            raise DfsError(errno.EBADF, "DFS client is not connected", self.root_dir)
        rel = normalize_path(path).lstrip("/")
        return os.path.join(self.root_dir, rel) if rel else self.root_dir

    def stat(self, path: str):
        local = self._local(path)
        if os.path.isdir(local):
            return "dir", 0
        if os.path.isfile(local):
            return "file", os.path.getsize(local)
        return None, 0

    def listdir(self, path: str):
        return os.listdir(self._local(path))

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        local = self._local(path)
        if local == self.root_dir:
            if exist_ok:
                return
            raise FileExistsError(errno.EEXIST, "exists", normalize_path(path))
        if parents:
            if os.path.isdir(local) and not exist_ok:
                raise FileExistsError(errno.EEXIST, "exists", normalize_path(path))
            os.makedirs(local, exist_ok=exist_ok)
            return
        try:
            os.mkdir(local)
        except FileExistsError:
            if not exist_ok or not os.path.isdir(local):
                raise

    def remove(self, path: str, recursive: bool = False) -> None:
        local = self._local(path)
        if local == self.root_dir:
            raise DfsError(errno.EBUSY, "refusing to remove the container root", "/")
        if os.path.isdir(local) and not os.path.islink(local):
            if recursive:
                shutil.rmtree(local)
            else:
                os.rmdir(local)
        else:
            os.remove(local)

    def read(self, path: str, offset: int = 0, length=None) -> bytes:
        with self.open_read(path) as f:
            remaining = max(0, f.size() - offset)
            n = remaining if length is None else max(0, min(int(length), remaining))
            buf = bytearray(n)
            if n:
                f.seek(offset)
                f.readinto(buf)
            return bytes(buf)

    def readinto(self, path: str, buf, offset: int = 0) -> int:
        with self.open_read(path) as f:
            f.seek(offset)
            return f.readinto(buf)

    def write(self, path: str, data) -> None:
        with self.open_write(path) as w:
            w.write(data)

    def open_write(self, path: str) -> _DirWriter:
        return _DirWriter(normalize_path(path), self._local(path))

    def open_read(self, path: str) -> _DirFile:
        local = self._local(path)
        if os.path.isdir(local):
            raise IsADirectoryError(errno.EISDIR, "is a directory", normalize_path(path))
        return _DirFile(local, "r")

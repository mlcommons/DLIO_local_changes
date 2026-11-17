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
DAOS File System (DFS) access through libdfs and ctypes.

libdfs.so is loaded the first time a ``DfsClient`` without an injected
library connects, so importing this module never requires DAOS.

Only these libdfs symbols are used; their declarations are identical in the
DAOS 2.6 and 2.8 public headers (``src/include/daos_fs.h``):

    dfs_init, dfs_fini, dfs_connect, dfs_disconnect, dfs_lookup,
    dfs_release, dfs_get_size, dfs_read, dfs_write, dfs_open, dfs_mkdir,
    dfs_iterate, dfs_remove

``dfs_readdir`` and ``dfs_stat`` are deliberately avoided: they fill the
platform ``struct dirent`` / ``struct stat``.  Names come from the
``dfs_iterate`` callback, the entry type from the ``mode`` that
``dfs_lookup`` returns, and the size from ``dfs_get_size``.

Ownership: one connected client per (process, pool, container, system),
obtained through ``get_client()``.  Readers, storage objects and
checkpointing in the same process share it.  ``close_all()`` disconnects
every client owned by the calling process.
"""

import contextlib
import ctypes
import errno
import io
import logging
import multiprocessing.util
import os
import posixpath
import stat
import threading

import numpy as np

__all__ = [
    "DfsError",
    "DfsClient",
    "DfsFile",
    "DfsWriter",
    "get_client",
    "get_worker_client",
    "close_all",
    "stand_in_dir",
]

_LIBDFS_NAME = "libdfs.so"
TEST_BACKING_DIR_ENV = "DLIO_DAOS_TEST_BACKING_DIR"
TEST_ALLOW_STAND_IN_ENV = "DLIO_DAOS_TEST_ALLOW_STAND_IN"

_FILE_MODE = stat.S_IFREG | 0o644
_DIR_MODE = 0o755
_ITERATE_BATCH = 1024
_DFS_MAX_NAME = 255
_DAOS_ANCHOR_TYPE_EOF = 3
_DAOS_ANCHOR_BUF_MAX = 104
_CLOSE_WAIT_SECONDS = 300.0
_EXIT_PRIORITY = 10

_c_int = ctypes.c_int
_dfs_p = ctypes.c_void_p
_obj_p = ctypes.c_void_p
_mode_t = ctypes.c_uint32
_daos_size_t = ctypes.c_uint64
_daos_off_t = ctypes.c_uint64
_daos_oclass_id_t = ctypes.c_uint32


class _DIov(ctypes.Structure):
    """d_iov_t (gurt/types.h)."""
    _fields_ = [
        ("iov_buf", ctypes.c_void_p),
        ("iov_buf_len", ctypes.c_size_t),
        ("iov_len", ctypes.c_size_t),
    ]


class _DSgList(ctypes.Structure):
    """d_sg_list_t (gurt/types.h)."""
    _fields_ = [
        ("sg_nr", ctypes.c_uint32),
        ("sg_nr_out", ctypes.c_uint32),
        ("sg_iovs", ctypes.POINTER(_DIov)),
    ]


class _DaosAnchor(ctypes.Structure):
    """daos_anchor_t (daos_types.h).  Zero-initialised means 'start'."""
    _fields_ = [
        ("da_type", ctypes.c_uint16),
        ("da_shard", ctypes.c_uint16),
        ("da_flags", ctypes.c_uint32),
        ("da_sub_anchors", ctypes.c_uint64),
        ("da_buf", ctypes.c_uint8 * _DAOS_ANCHOR_BUF_MAX),
    ]


# typedef int (*dfs_filler_cb_t)(dfs_t *, dfs_obj_t *, const char name[], void *arg)
_FILLER_CB = ctypes.CFUNCTYPE(_c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_char_p, ctypes.c_void_p)


_lib = None
_lib_lock = threading.Lock()


def _load_lib():
    """Load libdfs.so once per address space and declare prototypes."""
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        lib = ctypes.CDLL(_LIBDFS_NAME)

        def proto(name, restype, argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes

        proto("dfs_init", _c_int, [])
        proto("dfs_fini", _c_int, [])
        # int dfs_connect(const char *pool, const char *sys, const char *cont,
        #                 int flags, dfs_attr_t *attr, dfs_t **dfs)
        proto("dfs_connect", _c_int,
              [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, _c_int,
               ctypes.c_void_p, ctypes.POINTER(_dfs_p)])
        # int dfs_disconnect(dfs_t *dfs)
        proto("dfs_disconnect", _c_int, [_dfs_p])
        # int dfs_lookup(dfs_t *dfs, const char *path, int flags,
        #                dfs_obj_t **obj, mode_t *mode, struct stat *stbuf)
        proto("dfs_lookup", _c_int,
              [_dfs_p, ctypes.c_char_p, _c_int, ctypes.POINTER(_obj_p),
               ctypes.POINTER(_mode_t), ctypes.c_void_p])
        # int dfs_release(dfs_obj_t *obj)
        proto("dfs_release", _c_int, [_obj_p])
        # int dfs_get_size(dfs_t *dfs, dfs_obj_t *obj, daos_size_t *size)
        proto("dfs_get_size", _c_int,
              [_dfs_p, _obj_p, ctypes.POINTER(_daos_size_t)])
        # int dfs_read(dfs_t *dfs, dfs_obj_t *obj, d_sg_list_t *sgl,
        #              daos_off_t off, daos_size_t *read_size, daos_event_t *ev)
        proto("dfs_read", _c_int,
              [_dfs_p, _obj_p, ctypes.POINTER(_DSgList), _daos_off_t,
               ctypes.POINTER(_daos_size_t), ctypes.c_void_p])
        # int dfs_write(dfs_t *dfs, dfs_obj_t *obj, d_sg_list_t *sgl,
        #               daos_off_t off, daos_event_t *ev)
        proto("dfs_write", _c_int,
              [_dfs_p, _obj_p, ctypes.POINTER(_DSgList), _daos_off_t,
               ctypes.c_void_p])
        # int dfs_open(dfs_t *dfs, dfs_obj_t *parent, const char *name,
        #              mode_t mode, int flags, daos_oclass_id_t cid,
        #              daos_size_t chunk_size, const char *value,
        #              dfs_obj_t **obj)
        proto("dfs_open", _c_int,
              [_dfs_p, _obj_p, ctypes.c_char_p, _mode_t, _c_int,
               _daos_oclass_id_t, _daos_size_t, ctypes.c_char_p,
               ctypes.POINTER(_obj_p)])
        # int dfs_mkdir(dfs_t *dfs, dfs_obj_t *parent, const char *name,
        #               mode_t mode, daos_oclass_id_t cid)
        proto("dfs_mkdir", _c_int,
              [_dfs_p, _obj_p, ctypes.c_char_p, _mode_t, _daos_oclass_id_t])
        # int dfs_iterate(dfs_t *dfs, dfs_obj_t *obj, daos_anchor_t *anchor,
        #                 uint32_t *nr, size_t size, dfs_filler_cb_t op,
        #                 void *arg)
        proto("dfs_iterate", _c_int,
              [_dfs_p, _obj_p, ctypes.POINTER(_DaosAnchor),
               ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, _FILLER_CB,
               ctypes.c_void_p])
        # int dfs_remove(dfs_t *dfs, dfs_obj_t *parent, const char *name,
        #                bool force, daos_obj_id_t *oid)
        proto("dfs_remove", _c_int,
              [_dfs_p, _obj_p, ctypes.c_char_p, ctypes.c_bool,
               ctypes.c_void_p])
        _lib = lib
        return _lib


# -- Errors -----------------------------------------------------------------


class DfsError(OSError):
    """A libdfs call failed; ``errno`` is the DFS return code."""


class DfsFileNotFoundError(DfsError, FileNotFoundError):
    pass


class DfsFileExistsError(DfsError, FileExistsError):
    pass


class DfsNotADirectoryError(DfsError, NotADirectoryError):
    pass


class DfsIsADirectoryError(DfsError, IsADirectoryError):
    pass


class DfsPermissionError(DfsError, PermissionError):
    pass


_ERRNO_CLASSES = {
    errno.ENOENT: DfsFileNotFoundError,
    errno.EEXIST: DfsFileExistsError,
    errno.ENOTDIR: DfsNotADirectoryError,
    errno.EISDIR: DfsIsADirectoryError,
    errno.EACCES: DfsPermissionError,
    errno.EPERM: DfsPermissionError,
}


def _dfs_error(rc, func, path=None):
    """Exception for a DFS call that returned the errno ``rc``."""
    cls = _ERRNO_CLASSES.get(rc, DfsError)
    try:
        reason = os.strerror(rc)
    except ValueError:
        reason = f"error {rc}"
    return cls(rc, f"{func}: {reason}", path)


# -- Helpers ----------------------------------------------------------------


def normalize_path(path) -> str:
    """Absolute, normalised container path (``""`` and ``"."`` mean ``"/"``)."""
    if path is None:
        return "/"
    p = os.fsdecode(path)
    p = "/" + p.lstrip("/")
    p = posixpath.normpath(p)
    if p.startswith("//"):
        p = "/" + p.lstrip("/")
    return p


def _buffer(data, writable=False):
    """Return ``(keepalive, address, nbytes)`` for a contiguous byte buffer.

    No copy is made.  ``keepalive`` must stay referenced for the duration of
    the foreign call that uses ``address``.
    """
    if hasattr(data, "getbuffer"):
        data = data.getbuffer()
    mv = memoryview(data)
    if not mv.c_contiguous:
        raise BufferError("DFS I/O requires a C-contiguous buffer")
    if mv.format != "B" or mv.ndim != 1:
        mv = mv.cast("B")
    if writable and mv.readonly:
        raise TypeError("readinto() requires a writable buffer")
    nbytes = mv.nbytes
    if nbytes == 0:
        return mv, None, 0
    arr = np.frombuffer(mv, dtype=np.uint8)
    return (mv, arr), arr.ctypes.data, nbytes


# -- File handles -----------------------------------------------------------


class DfsFile(io.RawIOBase):
    """Read-only, seekable view of one DFS file.

    The size is taken once, when the file is opened.  A read that returns
    fewer bytes than that size allows (the file shrank) raises ``OSError``.
    """

    def __init__(self, client, path, obj, size):
        super().__init__()
        self._client = client
        self._path = path
        self._obj = obj
        self._size = int(size)
        self._pos = 0

    @property
    def name(self):
        return self._path

    def readable(self):
        return True

    def seekable(self):
        return True

    def size(self):
        return self._size

    def tell(self):
        self._checkClosed()
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence ({whence})")
        if pos < 0:
            raise ValueError(f"negative seek position {pos}")
        self._pos = pos
        return pos

    def readinto(self, b):
        self._checkClosed()
        keep, addr, cap = _buffer(b, writable=True)
        n = max(0, min(cap, self._size - self._pos))
        if n == 0:
            return 0
        self._client._read_obj(self._obj, addr, n, self._pos, self._path)
        del keep
        self._pos += n
        return n

    def readall(self):
        n = max(0, self._size - self._pos)
        buf = bytearray(n)
        if n:
            self.readinto(buf)
        return bytes(buf)

    def close(self):
        if not self.closed:
            obj, self._obj = self._obj, None
            try:
                if obj is not None:
                    self._client._release_handle(self, obj)
            finally:
                super().close()


class DfsWriter:
    """Sequential writer: each ``write()`` lands at the running offset."""

    def __init__(self, client, path, obj):
        self._client = client
        self._path = path
        self._obj = obj
        self._offset = 0

    @property
    def name(self):
        return self._path

    @property
    def closed(self):
        return self._obj is None

    def tell(self):
        return self._offset

    def write(self, data) -> int:
        if self._obj is None:
            raise ValueError("write to closed DfsWriter")
        keep, addr, n = _buffer(data)
        if n:
            self._client._write_obj(self._obj, addr, n, self._offset, self._path)
        del keep
        self._offset += n
        return n

    def close(self):
        obj, self._obj = self._obj, None
        if obj is not None:
            self._client._release_handle(self, obj)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# -- Client -----------------------------------------------------------------


class DfsClient:
    """A DFS mount of one pool/container, usable from many threads.

    ``lib`` is any object exposing the ``dfs_*`` callables; ``None`` loads
    libdfs.so on ``connect()``.
    """

    def __init__(self, pool: str, container: str, sys_name=None, lib=None):
        if not pool:
            raise ValueError("DAOS pool is required")
        if not container:
            raise ValueError("DAOS container is required")
        self.pool = str(pool)
        self.container = str(container)
        self.sys_name = str(sys_name) if sys_name else None
        self._injected_lib = lib
        self._lib = None
        self._dfs = None
        self._pid = None
        self._initialized = False
        self._closing = False
        self._inflight = 0
        self._cond = threading.Condition(threading.Lock())
        self._connect_lock = threading.Lock()
        self._dirs = {}
        self._retired = []
        self._handles = set()

    def __repr__(self):
        return (f"DfsClient(pool={self.pool!r}, container={self.container!r}, "
                f"sys_name={self.sys_name!r})")

    @property
    def connected(self):
        return self._dfs is not None and self._pid == os.getpid()

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        """dfs_init + dfs_connect(O_RDWR).  Idempotent."""
        with self._connect_lock:
            if self.connected:
                return
            if self._dfs is not None and self._pid != os.getpid():
                self._forget()
            lib = self._injected_lib if self._injected_lib is not None else _load_lib()
            rc = lib.dfs_init()
            if rc != 0:
                raise _dfs_error(rc, "dfs_init")
            with self._cond:
                self._lib = lib
                self._initialized = True
                self._pid = os.getpid()
            dfs = _dfs_p()
            sys_name = self.sys_name.encode() if self.sys_name else None
            rc = lib.dfs_connect(self.pool.encode(), sys_name,
                                 self.container.encode(), os.O_RDWR, None,
                                 ctypes.pointer(dfs))
            if rc == 0 and not dfs.value:
                rc = errno.EIO
            if rc != 0:
                self._finish_partial()
                raise _dfs_error(rc, "dfs_connect",
                                 f"{self.pool}/{self.container}")
            with self._cond:
                self._dfs = dfs
                self._closing = False

    def _finish_partial(self):
        with self._cond:
            initialized, self._initialized = self._initialized, False
            lib = self._lib
            self._pid = None
        if initialized and lib is not None:
            lib.dfs_fini()

    def _forget(self):
        """Drop handles inherited from another process without using them."""
        with self._cond:
            self._dfs = None
            self._dirs = {}
            self._retired = []
            self._handles = set()
            self._initialized = False
            self._inflight = 0
            self._pid = None

    def close(self) -> None:
        """Release open objects, dfs_disconnect, dfs_fini.  Idempotent.

        Waits for in-flight operations of other threads to finish first.
        """
        with self._connect_lock:
            if self._pid is not None and self._pid != os.getpid():
                self._forget()
                return
            with self._cond:
                if not self._initialized and self._dfs is None:
                    return
                self._closing = True
                if not self._cond.wait_for(lambda: self._inflight == 0,
                                           timeout=_CLOSE_WAIT_SECONDS):
                    logging.warning(
                        f"{self!r}: {self._inflight} operation(s) still running "
                        f"after {_CLOSE_WAIT_SECONDS:.0f}s; leaving the DFS mount "
                        "open instead of disconnecting under them")
                    self._dfs = None
                    self._dirs = {}
                    self._retired = []
                    self._handles = set()
                    self._initialized = False
                    self._pid = None
                    return
                lib = self._lib
                dfs = self._dfs
                handles = list(self._handles)
                dirs = list(self._dirs.items()) + self._retired
                self._handles = set()
                self._dirs = {}
                self._retired = []
                self._dfs = None
                initialized, self._initialized = self._initialized, False
                self._pid = None

            first_err = None
            for handle in handles:
                obj = getattr(handle, "_obj", None)
                if obj is None:
                    continue
                handle._obj = None
                rc = lib.dfs_release(obj)
                if rc != 0 and first_err is None:
                    first_err = _dfs_error(rc, "dfs_release", handle.name)
                if isinstance(handle, io.IOBase):
                    try:
                        io.RawIOBase.close(handle)
                    except Exception:
                        pass
            for (path, _), obj in dirs:
                rc = lib.dfs_release(obj)
                if rc != 0 and first_err is None:
                    first_err = _dfs_error(rc, "dfs_release", path)
            if dfs is not None:
                rc = lib.dfs_disconnect(dfs)
                if rc != 0 and first_err is None:
                    first_err = _dfs_error(rc, "dfs_disconnect",
                                           f"{self.pool}/{self.container}")
            if initialized:
                rc = lib.dfs_fini()
                if rc != 0 and first_err is None:
                    first_err = _dfs_error(rc, "dfs_fini")
            if first_err is not None:
                raise first_err

    @contextlib.contextmanager
    def _op(self):
        with self._cond:
            if self._dfs is None or self._closing:
                raise DfsError(errno.EBADF, "DFS client is not connected",
                               f"{self.pool}/{self.container}")
            if self._pid != os.getpid():
                raise DfsError(errno.EBADF,
                               "DFS client belongs to another process",
                               f"{self.pool}/{self.container}")
            self._inflight += 1
            lib, dfs = self._lib, self._dfs
        try:
            yield lib, dfs
        finally:
            with self._cond:
                self._inflight -= 1
                if self._inflight == 0:
                    self._cond.notify_all()

    # -- object handles ----------------------------------------------------

    def _track(self, handle):
        with self._cond:
            self._handles.add(handle)

    def _release_handle(self, handle, obj):
        with self._cond:
            if handle not in self._handles:
                return
            self._handles.discard(handle)
            lib = self._lib
            self._inflight += 1
        try:
            rc = lib.dfs_release(obj)
        finally:
            with self._cond:
                self._inflight -= 1
                if self._inflight == 0:
                    self._cond.notify_all()
        if rc != 0:
            raise _dfs_error(rc, "dfs_release", handle.name)

    def _lookup(self, lib, dfs, path, flags):
        """dfs_lookup -> (obj, mode).  The caller releases obj."""
        obj = _obj_p()
        mode = _mode_t(0)
        rc = lib.dfs_lookup(dfs, os.fsencode(path), flags, ctypes.pointer(obj),
                            ctypes.pointer(mode), None)
        if rc != 0:
            raise _dfs_error(rc, "dfs_lookup", path)
        return obj, mode.value

    def _dir(self, lib, dfs, path, flags):
        """Cached directory object for ``path`` opened with ``flags``."""
        key = (path, flags)
        with self._cond:
            obj = self._dirs.get(key)
        if obj is not None:
            return obj
        obj, mode = self._lookup(lib, dfs, path, flags)
        if not stat.S_ISDIR(mode):
            lib.dfs_release(obj)
            raise _dfs_error(errno.ENOTDIR, "dfs_lookup", path)
        with self._cond:
            cached = self._dirs.get(key)
            if cached is None:
                self._dirs[key] = obj
                return obj
        lib.dfs_release(obj)
        return cached

    def _retire_dirs_under(self, path):
        """Stop using cached directory objects at or below ``path``.

        They are released by ``close()``, not here, because another thread
        may still be passing one of them to libdfs.
        """
        prefix = path.rstrip("/") + "/"
        with self._cond:
            for key in [k for k in self._dirs if k[0] == path or k[0].startswith(prefix)]:
                self._retired.append((key, self._dirs.pop(key)))

    def _parent_and_name(self, path):
        path = normalize_path(path)
        parent, name = posixpath.split(path)
        return path, parent, name

    def _open_file(self, lib, dfs, path, mode, flags):
        path, parent, name = self._parent_and_name(path)
        if not name:
            raise _dfs_error(errno.EISDIR, "dfs_open", path)
        dir_flags = os.O_RDWR if flags & os.O_CREAT else os.O_RDONLY
        pobj = self._dir(lib, dfs, parent, dir_flags)
        obj = _obj_p()
        rc = lib.dfs_open(dfs, pobj, os.fsencode(name), mode, flags, 0, 0, None,
                          ctypes.pointer(obj))
        if rc != 0:
            raise _dfs_error(rc, "dfs_open", path)
        return path, obj

    def _get_size(self, lib, dfs, obj, path):
        size = _daos_size_t(0)
        rc = lib.dfs_get_size(dfs, obj, ctypes.pointer(size))
        if rc != 0:
            raise _dfs_error(rc, "dfs_get_size", path)
        return size.value

    def _read_obj(self, obj, addr, n, offset, path):
        with self._op() as (lib, dfs):
            iov = _DIov(addr, n, n)
            sgl = _DSgList(1, 0, ctypes.pointer(iov))
            got = _daos_size_t(0)
            rc = lib.dfs_read(dfs, obj, ctypes.pointer(sgl), offset,
                              ctypes.pointer(got), None)
            if rc != 0:
                raise _dfs_error(rc, "dfs_read", path)
            if got.value != n:
                raise DfsError(errno.EIO,
                               f"dfs_read: short read, {got.value} of {n} "
                               f"bytes at offset {offset}", path)
        return n

    def _write_obj(self, obj, addr, n, offset, path):
        with self._op() as (lib, dfs):
            iov = _DIov(addr, n, n)
            sgl = _DSgList(1, 0, ctypes.pointer(iov))
            rc = lib.dfs_write(dfs, obj, ctypes.pointer(sgl), offset, None)
            if rc != 0:
                raise _dfs_error(rc, "dfs_write", path)

    # -- metadata ----------------------------------------------------------

    def stat(self, path: str):
        """``("file"|"dir"|None, size)``; size is 0 for directories."""
        path = normalize_path(path)
        with self._op() as (lib, dfs):
            try:
                obj, mode = self._lookup(lib, dfs, path, os.O_RDONLY)
            except (FileNotFoundError, NotADirectoryError):
                return None, 0
            try:
                if stat.S_ISDIR(mode):
                    return "dir", 0
                if stat.S_ISREG(mode):
                    return "file", self._get_size(lib, dfs, obj, path)
                return "file", 0
            finally:
                lib.dfs_release(obj)

    def listdir(self, path: str):
        """Names of the direct children of ``path`` (``dfs_iterate``)."""
        path = normalize_path(path)
        with self._op() as (lib, dfs):
            obj, mode = self._lookup(lib, dfs, path, os.O_RDONLY)
            try:
                if not stat.S_ISDIR(mode):
                    raise _dfs_error(errno.ENOTDIR, "dfs_iterate", path)
                return self._iterate(lib, dfs, obj, path)
            finally:
                lib.dfs_release(obj)

    def _iterate(self, lib, dfs, obj, path):
        names = []
        failures = []

        def _on_entry(_dfs, _obj, name, _arg):
            try:
                names.append(os.fsdecode(name))
                return 0
            except BaseException as exc:  # noqa: BLE001 - must not cross into C
                failures.append(exc)
                return errno.EIO

        callback = _FILLER_CB(_on_entry)
        anchor = _DaosAnchor()
        size = _ITERATE_BATCH * (_DFS_MAX_NAME + 1)
        while anchor.da_type != _DAOS_ANCHOR_TYPE_EOF:
            nr = ctypes.c_uint32(_ITERATE_BATCH)
            rc = lib.dfs_iterate(dfs, obj, ctypes.pointer(anchor),
                                 ctypes.pointer(nr), size, callback, None)
            if failures:
                raise DfsError(errno.EIO,
                               f"dfs_iterate: entry callback failed: "
                               f"{failures[0]!r}", path) from failures[0]
            if rc != 0:
                raise _dfs_error(rc, "dfs_iterate", path)
        del callback
        return names

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        path, parent, name = self._parent_and_name(path)
        if not name:
            if exist_ok:
                return
            raise _dfs_error(errno.EEXIST, "dfs_mkdir", path)
        with self._op() as (lib, dfs):
            try:
                pobj = self._dir(lib, dfs, parent, os.O_RDWR)
            except FileNotFoundError:
                if not parents:
                    raise
                pobj = None
        if pobj is None:
            self.mkdir(parent, parents=True, exist_ok=True)
            with self._op() as (lib, dfs):
                pobj = self._dir(lib, dfs, parent, os.O_RDWR)
        with self._op() as (lib, dfs):
            rc = lib.dfs_mkdir(dfs, pobj, os.fsencode(name), _DIR_MODE, 0)
        if rc == 0:
            return
        if rc != errno.EEXIST:
            raise _dfs_error(rc, "dfs_mkdir", path)
        if not exist_ok:
            raise _dfs_error(rc, "dfs_mkdir", path)
        kind, _ = self.stat(path)
        if kind != "dir":
            raise _dfs_error(errno.EEXIST, "dfs_mkdir: exists and is not a directory", path)

    def remove(self, path: str, recursive: bool = False) -> None:
        """``dfs_remove(force=recursive)``."""
        path, parent, name = self._parent_and_name(path)
        if not name:
            raise _dfs_error(errno.EBUSY, "dfs_remove: refusing to remove the container root", path)
        with self._op() as (lib, dfs):
            pobj = self._dir(lib, dfs, parent, os.O_RDWR)
            rc = lib.dfs_remove(dfs, pobj, os.fsencode(name), bool(recursive), None)
            self._retire_dirs_under(path)
            if rc != 0:
                raise _dfs_error(rc, "dfs_remove", path)

    # -- data --------------------------------------------------------------

    def open_read(self, path: str) -> DfsFile:
        with self._op() as (lib, dfs):
            path, obj = self._open_file(lib, dfs, path, stat.S_IFREG, os.O_RDONLY)
            try:
                size = self._get_size(lib, dfs, obj, path)
            except BaseException:
                lib.dfs_release(obj)
                raise
            handle = DfsFile(self, path, obj, size)
            self._track(handle)
            return handle

    def open_write(self, path: str) -> DfsWriter:
        with self._op() as (lib, dfs):
            path, obj = self._open_file(lib, dfs, path, _FILE_MODE,
                                        os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            handle = DfsWriter(self, path, obj)
            self._track(handle)
            return handle

    def read(self, path: str, offset: int = 0, length=None) -> bytes:
        """Whole file, or ``length`` bytes from ``offset`` (clamped to EOF)."""
        with self.open_read(path) as f:
            remaining = max(0, f.size() - offset)
            n = remaining if length is None else max(0, min(int(length), remaining))
            buf = bytearray(n)
            if n:
                f.seek(offset)
                f.readinto(buf)
            return bytes(buf)

    def readinto(self, path: str, buf, offset: int = 0) -> int:
        """Fill ``buf`` from ``offset``; returns the byte count (clamped to EOF)."""
        with self.open_read(path) as f:
            f.seek(offset)
            return f.readinto(buf)

    def write(self, path: str, data) -> None:
        """Create or truncate ``path`` and write ``data`` at offset 0."""
        with self.open_write(path) as w:
            w.write(data)


# -- Process-wide registry --------------------------------------------------

_registry = {}
_registry_lock = threading.Lock()
_exit_hook_pid = None


def stand_in_dir():
    """The backing directory of the test stand-in, or ``None`` for DAOS.

    The stand-in is used only when both ``DLIO_DAOS_TEST_BACKING_DIR`` and
    ``DLIO_DAOS_TEST_ALLOW_STAND_IN=1`` are set.  The first variable alone
    raises ``RuntimeError``, so a leftover test setting cannot turn a
    measurement into a local-directory run.
    """
    backing = os.environ.get(TEST_BACKING_DIR_ENV) or None
    if backing is None:
        return None
    if os.environ.get(TEST_ALLOW_STAND_IN_ENV) != "1":
        raise RuntimeError(
            f"{TEST_BACKING_DIR_ENV} is set but {TEST_ALLOW_STAND_IN_ENV}=1 is not. "
            f"The directory stand-in for storage_type daos is for DLIO's own tests "
            f"and needs both variables; unset {TEST_BACKING_DIR_ENV} to use DAOS.")
    return backing


def get_client(pool: str, container: str, sys_name=None):
    """The connected client of this process for (pool, container, sys_name).

    When ``stand_in_dir()`` selects the test stand-in, a directory-backed
    client rooted at ``<dir>/<pool>/<container>`` is returned instead and
    libdfs is never loaded.
    """
    if not pool:
        raise ValueError("DAOS pool is required (storage.storage_options.pool)")
    if not container:
        raise ValueError("DAOS container is required (storage.storage_options.container)")
    backing = stand_in_dir()
    key = (os.getpid(), str(pool), str(container), sys_name or None, backing)
    with _registry_lock:
        client = _registry.get(key)
        if client is not None:
            return client
        if backing is not None:
            from dlio_benchmark.storage.dfs_testing import DirBackedDfsClient
            client = DirBackedDfsClient(os.path.join(backing, str(pool), str(container)))
        else:
            client = DfsClient(pool, container, sys_name)
        client.connect()
        _registry[key] = client
        _register_exit_hook()
        return client


def _register_exit_hook():
    """Run ``close_all()`` once at this process's exit.  Needs ``_registry_lock``.

    multiprocessing's exit handling runs the hook after a worker's target
    returns or raises, and from ``atexit`` in any other process.  Clients
    already closed by an explicit ``close_all()`` are not closed again.
    """
    global _exit_hook_pid
    pid = os.getpid()
    if _exit_hook_pid != pid:
        multiprocessing.util.Finalize(None, close_all, exitpriority=_EXIT_PRIORITY)
        _exit_hook_pid = pid


def get_worker_client(pool: str, container: str, sys_name=None):
    """``get_client()`` for a DataLoader worker process.

    Same as ``get_client()``, which registers the exit hook for every
    process.
    """
    return get_client(pool, container, sys_name)


def close_all() -> None:
    """Close every client owned by this process.

    Clients inherited from another process are forgotten without calling
    into libdfs.  The first close error is raised after all were attempted.
    """
    pid = os.getpid()
    with _registry_lock:
        mine = [c for k, c in _registry.items() if k[0] == pid]
        _registry.clear()
    first_err = None
    for client in mine:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001
            if first_err is None:
                first_err = exc
    if first_err is not None:
        raise first_err

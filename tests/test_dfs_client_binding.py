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
Unit tests for the ctypes libdfs binding (``dlio_benchmark.storage.dfs_client``).

``FakeLib`` stands in for libdfs.so: it receives exactly the ctypes objects
the real ``DfsClient`` builds (pointers, SGLs, anchors, the ``dfs_iterate``
callback), keeps an in-memory tree, and records every call.  The real
binding code runs unchanged; only the shared library is replaced.
"""

import ctypes
import errno
import os
import posixpath
import stat
import threading
import time

import pytest

from dlio_benchmark.storage import dfs_client
from dlio_benchmark.storage.dfs_client import DfsClient, DfsError


ALLOWED_SYMBOLS = {
    "dfs_init", "dfs_fini", "dfs_connect", "dfs_disconnect", "dfs_lookup",
    "dfs_release", "dfs_get_size", "dfs_read", "dfs_write", "dfs_open",
    "dfs_mkdir", "dfs_iterate", "dfs_remove",
}

_DFS_HANDLE = 0x1000


def _v(x):
    """Integer value of a handle argument (c_void_p, int or None)."""
    if isinstance(x, ctypes.c_void_p):
        return x.value
    return x


class FakeLib:
    """In-memory libdfs with the calling conventions of the real library."""

    def __init__(self, iterate_batch=None, init_rc=0, connect_rc=0):
        self.calls = []
        self.tree = {"/": {"type": "dir"}}
        self.handles = {}
        self.next_handle = 0x2000
        self.iterate_batch = iterate_batch
        self.init_rc = init_rc
        self.connect_rc = connect_rc
        self.connect_flags = None
        self.open_calls = []
        self.short_reads = {}
        self.bad_name = False

    # -- helpers -------------------------------------------------------------

    def _record(self, name, *args):
        self.calls.append((name, args))

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)

    def _new_handle(self, path):
        h = self.next_handle
        self.next_handle += 1
        self.handles[h] = path
        return h

    def _resolve(self, path):
        """Return (entry or None, errno)."""
        if path == "/":
            return self.tree["/"], 0
        parts = [p for p in path.split("/") if p]
        cur = "/"
        for i, part in enumerate(parts):
            nxt = posixpath.join(cur, part)
            entry = self.tree.get(nxt)
            if entry is None:
                return None, errno.ENOENT
            if i < len(parts) - 1 and entry["type"] != "dir":
                return None, errno.ENOTDIR
            cur = nxt
        return self.tree[cur], 0

    def _children(self, path):
        prefix = path.rstrip("/") + "/"
        return sorted(
            p[len(prefix):] for p in self.tree
            if p != path and p.startswith(prefix) and "/" not in p[len(prefix):]
        )

    def add_file(self, path, data):
        parent = posixpath.dirname(path)
        assert self.tree[parent]["type"] == "dir"
        self.tree[path] = {"type": "file", "data": bytearray(data)}

    def add_dir(self, path):
        self.tree[path] = {"type": "dir"}

    # -- libdfs surface ------------------------------------------------------

    def dfs_init(self):
        self._record("dfs_init")
        return self.init_rc

    def dfs_fini(self):
        self._record("dfs_fini")
        return 0

    def dfs_connect(self, pool, sys_name, cont, flags, attr, dfs_pp):
        self._record("dfs_connect", pool, sys_name, cont, flags)
        self.connect_flags = flags
        if self.connect_rc:
            return self.connect_rc
        dfs_pp[0] = _DFS_HANDLE
        return 0

    def dfs_disconnect(self, dfs):
        self._record("dfs_disconnect", _v(dfs))
        return 0

    def dfs_lookup(self, dfs, path, flags, obj_pp, mode_p, stbuf):
        path = path.decode()
        self._record("dfs_lookup", path, flags)
        entry, rc = self._resolve(path)
        if rc:
            return rc
        obj_pp[0] = self._new_handle(path)
        if mode_p:
            mode_p[0] = (stat.S_IFDIR | 0o755) if entry["type"] == "dir" else (stat.S_IFREG | 0o644)
        return 0

    def dfs_release(self, obj):
        h = _v(obj)
        self._record("dfs_release", h)
        self.handles.pop(h, None)
        return 0

    def dfs_open(self, dfs, parent, name, mode, flags, cid, chunk_size, value, obj_pp):
        parent_path = self.handles[_v(parent)]
        path = posixpath.join(parent_path, name.decode())
        self._record("dfs_open", path, mode, flags)
        self.open_calls.append((path, mode, flags))
        entry = self.tree.get(path)
        if entry is None:
            if not flags & os.O_CREAT:
                return errno.ENOENT
            entry = self.tree[path] = {"type": "file", "data": bytearray()}
        elif entry["type"] != "file":
            return errno.EINVAL
        elif flags & os.O_TRUNC:
            entry["data"] = bytearray()
        obj_pp[0] = self._new_handle(path)
        return 0

    def dfs_get_size(self, dfs, obj, size_p):
        path = self.handles[_v(obj)]
        self._record("dfs_get_size", path)
        entry = self.tree[path]
        if entry["type"] != "file":
            return errno.EINVAL
        size_p[0] = len(entry["data"])
        return 0

    def dfs_read(self, dfs, obj, sgl_p, off, read_size_p, ev):
        path = self.handles[_v(obj)]
        sgl = sgl_p.contents
        iov = sgl.sg_iovs[0]
        self._record("dfs_read", path, off, iov.iov_buf_len)
        data = bytes(self.tree[path]["data"][off:off + iov.iov_buf_len])
        if path in self.short_reads:
            data = data[:self.short_reads[path]]
        ctypes.memmove(iov.iov_buf, data, len(data))
        read_size_p[0] = len(data)
        return 0

    def dfs_write(self, dfs, obj, sgl_p, off, ev):
        path = self.handles[_v(obj)]
        iov = sgl_p.contents.sg_iovs[0]
        payload = ctypes.string_at(iov.iov_buf, iov.iov_len)
        self._record("dfs_write", path, off, len(payload))
        buf = self.tree[path]["data"]
        if len(buf) < off:
            buf.extend(b"\0" * (off - len(buf)))
        buf[off:off + len(payload)] = payload
        return 0

    def dfs_mkdir(self, dfs, parent, name, mode, cid):
        path = posixpath.join(self.handles[_v(parent)], name.decode())
        self._record("dfs_mkdir", path, mode)
        if path in self.tree:
            return errno.EEXIST
        self.tree[path] = {"type": "dir"}
        return 0

    def dfs_iterate(self, dfs, obj, anchor_p, nr_p, size, op, arg):
        path = self.handles[_v(obj)]
        anchor = anchor_p.contents
        names = self._children(path)
        cursor = anchor.da_sub_anchors
        batch_size = self.iterate_batch or nr_p[0]
        batch = names[cursor:cursor + batch_size]
        self._record("dfs_iterate", path, cursor, len(batch))
        for n in batch:
            rc = op(_DFS_HANDLE, _v(obj), None if self.bad_name else n.encode(), None)
            if rc:
                return rc
        cursor += len(batch)
        anchor.da_sub_anchors = cursor
        if cursor >= len(names):
            anchor.da_type = 3
        nr_p[0] = len(batch)
        return 0

    def dfs_remove(self, dfs, parent, name, force, oid):
        path = posixpath.join(self.handles[_v(parent)], name.decode())
        self._record("dfs_remove", path, bool(force))
        entry = self.tree.get(path)
        if entry is None:
            return errno.ENOENT
        if entry["type"] == "dir" and self._children(path) and not force:
            return errno.ENOTEMPTY
        prefix = path + "/"
        for p in [p for p in self.tree if p == path or p.startswith(prefix)]:
            del self.tree[p]
        return 0


@pytest.fixture
def fake():
    return FakeLib()


@pytest.fixture
def client(fake):
    c = DfsClient("pool0", "cont0", lib=fake)
    c.connect()
    yield c
    c.close()


def test_connect_uses_o_rdwr_without_o_creat(fake):
    c = DfsClient("pool0", "cont0", lib=fake)
    c.connect()
    assert fake.connect_flags & os.O_ACCMODE == os.O_RDWR
    assert not fake.connect_flags & os.O_CREAT
    _, args = next(call for call in fake.calls if call[0] == "dfs_connect")
    assert args[0] == b"pool0" and args[2] == b"cont0"
    assert args[1] is None
    c.connect()
    assert fake.count("dfs_connect") == 1
    assert fake.count("dfs_init") == 1
    c.close()


def test_connect_passes_sys_name_as_bytes(fake):
    c = DfsClient("pool0", "cont0", sys_name="daos_server", lib=fake)
    c.connect()
    _, args = next(call for call in fake.calls if call[0] == "dfs_connect")
    assert args[1] == b"daos_server"
    c.close()


def test_connect_without_sys_name_passes_none(fake):
    c = DfsClient("pool0", "cont0", lib=fake)
    c.connect()
    _, args = next(call for call in fake.calls if call[0] == "dfs_connect")
    assert args[1] is None
    c.close()


def test_write_flags_and_mode(client, fake):
    client.write("/a.bin", b"hello")
    path, mode, flags = fake.open_calls[-1]
    assert path == "/a.bin"
    assert mode == stat.S_IFREG | 0o644
    assert flags == os.O_RDWR | os.O_CREAT | os.O_TRUNC
    assert bytes(fake.tree["/a.bin"]["data"]) == b"hello"


def test_write_accepts_bytesio_and_memoryview(client, fake):
    import io
    client.write("/b.bin", io.BytesIO(b"abc"))
    client.write("/c.bin", memoryview(b"xyz"))
    assert bytes(fake.tree["/b.bin"]["data"]) == b"abc"
    assert bytes(fake.tree["/c.bin"]["data"]) == b"xyz"


def test_read_range(client, fake):
    data = bytes(range(64))
    fake.add_file("/r.bin", data)
    assert client.read("/r.bin") == data
    assert client.read("/r.bin", 3, 5) == data[3:8]
    assert client.read("/r.bin", 60, 100) == data[60:]
    assert client.read("/r.bin", 100) == b""


def test_readinto_fills_buffer_without_copy(client, fake):
    data = bytes(range(32))
    fake.add_file("/x.bin", data)
    buf = bytearray(16)
    n = client.readinto("/x.bin", buf, offset=8)
    assert n == 16
    assert bytes(buf) == data[8:24]


def test_short_read_raises_oserror(client, fake):
    fake.add_file("/s.bin", b"x" * 100)
    fake.short_reads["/s.bin"] = 40
    with pytest.raises(OSError) as exc_info:
        client.read("/s.bin")
    assert exc_info.value.errno == errno.EIO
    buf = bytearray(100)
    with pytest.raises(OSError):
        client.readinto("/s.bin", buf)


def test_missing_path_stat_returns_none(client, fake):
    assert client.stat("/missing") == (None, 0)
    fake.add_file("/f", b"1234")
    assert client.stat("/f/below") == (None, 0)
    assert client.stat("/f") == ("file", 4)
    assert client.stat("/") == ("dir", 0)
    fake.add_dir("/d")
    assert client.stat("/d") == ("dir", 0)
    assert all(args[0] != "/d" for name, args in fake.calls if name == "dfs_get_size")


def test_read_missing_raises_filenotfound(client):
    with pytest.raises(FileNotFoundError):
        client.read("/nope.bin")
    with pytest.raises(FileNotFoundError):
        client.read("/no_dir/nope.bin")
    with pytest.raises(FileNotFoundError):
        client.open_read("/nope.bin")


def test_errors_are_dfs_errors(client):
    with pytest.raises(DfsError) as exc_info:
        client.read("/nope.bin")
    assert exc_info.value.errno == errno.ENOENT


def test_listdir_multi_batch_returns_each_name_once():
    fake = FakeLib(iterate_batch=3)
    fake.add_dir("/d")
    names = [f"f{i:02d}" for i in range(8)]
    for n in names:
        fake.add_file(f"/d/{n}", b"")
    c = DfsClient("pool0", "cont0", lib=fake)
    c.connect()
    try:
        listed = c.listdir("/d")
    finally:
        c.close()
    assert sorted(listed) == names
    assert len(listed) == len(set(listed))
    assert fake.count("dfs_iterate") == 3


def test_listdir_missing_and_not_a_directory(client, fake):
    with pytest.raises(FileNotFoundError):
        client.listdir("/missing")
    fake.add_file("/f", b"")
    with pytest.raises(NotADirectoryError):
        client.listdir("/f")


def test_iterate_callback_exception_becomes_error(client, fake):
    fake.add_dir("/d")
    fake.add_file("/d/a", b"")
    fake.bad_name = True
    with pytest.raises(DfsError) as exc_info:
        client.listdir("/d")
    assert exc_info.value.errno == errno.EIO


def test_mkdir_parents_and_exist_ok(client, fake):
    client.mkdir("/a/b/c")
    assert fake.tree["/a"]["type"] == "dir"
    assert fake.tree["/a/b/c"]["type"] == "dir"
    client.mkdir("/a/b/c", exist_ok=True)
    with pytest.raises(FileExistsError):
        client.mkdir("/a/b/c", exist_ok=False)
    with pytest.raises(FileNotFoundError):
        client.mkdir("/x/y", parents=False)
    fake.add_file("/a/file", b"")
    with pytest.raises(FileExistsError):
        client.mkdir("/a/file", exist_ok=True)


def test_remove_passes_force(client, fake):
    client.mkdir("/t/u")
    fake.add_file("/t/u/f", b"1")
    with pytest.raises(OSError):
        client.remove("/t", recursive=False)
    client.remove("/t", recursive=True)
    assert "/t" not in fake.tree
    assert ("dfs_remove", ("/t", True)) in fake.calls


def test_open_write_appends_at_running_offset(client, fake):
    with client.open_write("/ck.bin") as w:
        w.write(b"a" * 10)
        w.write(b"b" * 5)
        assert w.tell() == 15
    writes = [args for name, args in fake.calls if name == "dfs_write"]
    assert [(off, n) for _, off, n in writes] == [(0, 10), (10, 5)]
    truncating_opens = [c for c in fake.open_calls if c[0] == "/ck.bin"]
    assert len(truncating_opens) == 1
    assert bytes(fake.tree["/ck.bin"]["data"]) == b"a" * 10 + b"b" * 5


def test_open_read_is_seekable_rawio(client, fake):
    data = bytes(range(100))
    fake.add_file("/p.bin", data)
    with client.open_read("/p.bin") as f:
        assert f.seekable() and f.readable()
        assert f.size() == 100
        f.seek(-10, os.SEEK_END)
        assert f.tell() == 90
        assert f.read(4) == data[90:94]
        f.seek(0)
        assert f.read() == data
    released = fake.count("dfs_release")
    assert released >= 1


def test_close_after_partial_connect():
    fake = FakeLib(connect_rc=errno.ENOENT)
    c = DfsClient("pool0", "missing", lib=fake)
    with pytest.raises(FileNotFoundError):
        c.connect()
    c.close()
    c.close()
    assert fake.count("dfs_init") == 1
    assert fake.count("dfs_fini") == 1
    assert fake.count("dfs_disconnect") == 0


def test_close_after_failed_init():
    fake = FakeLib(init_rc=errno.EIO)
    c = DfsClient("pool0", "cont0", lib=fake)
    with pytest.raises(DfsError):
        c.connect()
    c.close()
    assert fake.count("dfs_fini") == 0
    assert fake.count("dfs_disconnect") == 0


def test_close_twice_is_noop(fake):
    c = DfsClient("pool0", "cont0", lib=fake)
    c.connect()
    fake.add_dir("/d")
    fake.add_file("/d/f", b"abc")
    c.read("/d/f")
    open_file = c.open_read("/d/f")
    c.close()
    c.close()
    assert fake.count("dfs_disconnect") == 1
    assert fake.count("dfs_fini") == 1
    assert not fake.handles, f"objects left open: {fake.handles}"
    assert open_file.closed
    open_file.close()
    with pytest.raises(OSError):
        c.read("/d/f")


def test_only_allowed_symbols_are_called(client, fake):
    client.mkdir("/a/b")
    client.write("/a/b/f", b"data")
    client.read("/a/b/f", 1, 2)
    client.stat("/a/b/f")
    client.listdir("/a/b")
    client.remove("/a", recursive=True)
    used = {name for name, _ in fake.calls}
    assert used <= ALLOWED_SYMBOLS, used - ALLOWED_SYMBOLS


def test_get_client_requires_pool_and_container():
    with pytest.raises(ValueError, match="pool"):
        dfs_client.get_client("", "c")
    with pytest.raises(ValueError, match="container"):
        dfs_client.get_client("p", None)


def test_get_client_is_per_pid_singleton(monkeypatch):
    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV, raising=False)
    monkeypatch.setattr(dfs_client, "_registry", {})
    libs = []

    def _fake_load():
        libs.append(FakeLib())
        return libs[-1]

    monkeypatch.setattr(dfs_client, "_load_lib", _fake_load)
    a = dfs_client.get_client("pool0", "cont0")
    b = dfs_client.get_client("pool0", "cont0")
    assert a is b
    other = dfs_client.get_client("pool0", "cont1")
    assert other is not a
    real_pid = os.getpid()
    with monkeypatch.context() as m:
        m.setattr(dfs_client.os, "getpid", lambda: real_pid + 100000)
        child = dfs_client.get_client("pool0", "cont0")
        assert child is not a
        dfs_client.close_all()
    assert libs[0].count("dfs_disconnect") == 0
    assert dfs_client._registry == {}


def test_close_all_closes_this_process_clients(monkeypatch):
    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV, raising=False)
    monkeypatch.setattr(dfs_client, "_registry", {})
    fake = FakeLib()
    monkeypatch.setattr(dfs_client, "_load_lib", lambda: fake)
    dfs_client.get_client("pool0", "cont0")
    dfs_client.close_all()
    dfs_client.close_all()
    assert fake.count("dfs_disconnect") == 1
    assert fake.count("dfs_fini") == 1


def test_backing_dir_env_selects_dir_client(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_testing import DirBackedDfsClient

    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dfs_client, "_registry", {})

    def _no_libdfs():
        raise AssertionError("libdfs must not be loaded with a backing dir")

    monkeypatch.setattr(dfs_client, "_load_lib", _no_libdfs)
    c = dfs_client.get_client("pool0", "cont0")
    try:
        assert isinstance(c, DirBackedDfsClient)
        assert c.root_dir == os.path.join(str(tmp_path), "pool0", "cont0")
        c.mkdir("/d/e")
        c.write("/d/e/f.bin", b"0123456789")
        assert (tmp_path / "pool0" / "cont0" / "d" / "e" / "f.bin").read_bytes() == b"0123456789"
        assert c.read("/d/e/f.bin", 2, 3) == b"234"
        assert c.stat("/d/e/f.bin") == ("file", 10)
        assert c.stat("/d") == ("dir", 0)
        assert c.stat("/nope") == (None, 0)
        assert c.listdir("/d/e") == ["f.bin"]
        with pytest.raises(FileNotFoundError):
            c.read("/nope")
    finally:
        dfs_client.close_all()


def test_close_all_waits_for_running_read_and_rejects_new_ones(monkeypatch):
    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV, raising=False)
    monkeypatch.setattr(dfs_client, "_registry", {})
    fake = FakeLib()
    fake.add_dir("/d")
    fake.add_file("/d/f", b"x" * 64)
    events = []
    entered = threading.Event()
    release = threading.Event()
    real_read = fake.dfs_read
    real_disconnect = fake.dfs_disconnect

    def _blocking_read(*a):
        entered.set()
        assert release.wait(30)
        rc = real_read(*a)
        events.append("read_done")
        return rc

    def _disconnect(*a):
        events.append("disconnect")
        return real_disconnect(*a)

    fake.dfs_read = _blocking_read
    fake.dfs_disconnect = _disconnect
    monkeypatch.setattr(dfs_client, "_load_lib", lambda: fake)
    client = dfs_client.get_client("pool0", "cont0")

    result = {}

    def _read():
        try:
            result["data"] = client.read("/d/f")
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    assert entered.wait(10)
    closer = threading.Thread(target=dfs_client.close_all, daemon=True)
    closer.start()
    deadline = time.monotonic() + 10
    while not client._closing and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client._closing, "close_all did not start closing the client"

    with pytest.raises(OSError, match="not connected"):
        client.read("/d/f")
    closer.join(timeout=0.2)
    assert closer.is_alive(), "close_all returned while a read was running"
    assert events == []

    release.set()
    reader.join(timeout=10)
    closer.join(timeout=10)
    assert not closer.is_alive()
    assert result == {"data": b"x" * 64}
    assert events == ["read_done", "disconnect"]
    assert fake.count("dfs_fini") == 1


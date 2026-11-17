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
Tests for the ``storage_type: daos`` checkpoint backend and its selection
in ``PyTorchCheckpointing._get_streaming()``.  libdfs is never loaded.
"""

import builtins
import io
import os
import sys
import threading
import types
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from dlio_benchmark.checkpointing import dfs_streaming_checkpointing as ckpt_mod
from dlio_benchmark.checkpointing.dfs_streaming_checkpointing import DfsStreamingCheckpointing
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.storage import dfs_client
from dlio_benchmark.storage.dfs_testing import DirBackedDfsClient

MiB = 1024 * 1024


class _RecordingWriter:
    def __init__(self, log, fail_at=None):
        self._log = log
        self._offset = 0
        self._writes = 0
        self._fail_at = fail_at
        self.closed = False

    def write(self, data):
        self._writes += 1
        if self._writes == self._fail_at:
            raise OSError("injected write failure")
        n = memoryview(data).nbytes
        self._log.append(("write", self._offset, n, bytes(memoryview(data)[:8])))
        self._offset += n
        return n

    def close(self):
        self.closed = True
        self._log.append(("close",))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _RecordingFile:
    def __init__(self, log, size):
        self._log = log
        self._size = size
        self._pos = 0
        self.closed = False

    def size(self):
        return self._size

    def seek(self, offset, whence=io.SEEK_SET):
        assert whence == io.SEEK_SET
        self._pos = offset
        return offset

    def readinto(self, b):
        n = max(0, min(memoryview(b).nbytes, self._size - self._pos))
        self._log.append(("read", self._pos, n))
        self._pos += n
        return n

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _RecordingClient:
    def __init__(self, size=0, fail_write_at=None):
        self.log = []
        self.size = size
        self.fail_write_at = fail_write_at

    def open_write(self, path):
        self.log.append(("open_write", path))
        return _RecordingWriter(self.log, self.fail_write_at)

    def open_read(self, path):
        self.log.append(("open_read", path))
        return _RecordingFile(self.log, self.size)

    def mkdir(self, path, parents=True, exist_ok=True):
        self.log.append(("mkdir", path))


# -- Selection -----------------------------------------------------------------


def _fake_mlpstorage(monkeypatch):
    class _MlpStreaming:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    pkg = types.ModuleType("mlpstorage_py")
    sub = types.ModuleType("mlpstorage_py.checkpointing")
    sub.StreamingCheckpointing = _MlpStreaming
    pkg.checkpointing = sub
    monkeypatch.setitem(sys.modules, "mlpstorage_py", pkg)
    monkeypatch.setitem(sys.modules, "mlpstorage_py.checkpointing", sub)
    return _MlpStreaming


def _init_mpi():
    from dlio_benchmark.common.enumerations import MPIState
    from dlio_benchmark.utils.utility import DLIOMPI

    if DLIOMPI.get_instance().mpi_state == MPIState.CHILD_INITIALIZED:
        DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()


def test_get_streaming_selects_dfs_for_daos(monkeypatch):
    pytest.importorskip("torch")
    # Importing the checkpointing module reaches framework.py, which calls
    # DLIOMPI.comm() at import time.
    _init_mpi()
    from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing

    mlp_cls = _fake_mlpstorage(monkeypatch)

    daos_self = SimpleNamespace(args=SimpleNamespace(storage_type=StorageType.DAOS))
    streaming = PyTorchCheckpointing._get_streaming(daos_self)
    assert isinstance(streaming, DfsStreamingCheckpointing)
    assert streaming.chunk_size == 32 * MiB
    # Same parameters as StreamingCheckpointing for file / direct_fs.
    assert streaming.num_buffers == 4
    assert streaming.num_parallel_readers == 4
    assert streaming.read_chunk_size == 4 * 32 * MiB
    assert PyTorchCheckpointing._get_streaming(daos_self) is streaming

    local_self = SimpleNamespace(args=SimpleNamespace(storage_type=StorageType.LOCAL_FS))
    assert isinstance(PyTorchCheckpointing._get_streaming(local_self), mlp_cls)


# -- save / load ---------------------------------------------------------------


class _FakeGenerator:
    """dgen-py ``Generator`` stand-in: fills with a counter, may fail."""

    def __init__(self, fail_at=None, short=False, **kwargs):
        self.kwargs = kwargs
        self.calls = 0
        self.fail_at = fail_at
        self.short = short

    def fill_chunk(self, buf):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ValueError("injected fill failure")
        view = np.frombuffer(buf, dtype=np.uint8)
        view[:] = self.calls % 251
        return view.nbytes - 1 if self.short else view.nbytes


def _fake_dgen(monkeypatch, **kwargs):
    made = []

    def _generator(**gen_kwargs):
        made.append(_FakeGenerator(**kwargs, **gen_kwargs))
        return made[-1]

    monkeypatch.setattr(ckpt_mod, "_dgen_py", SimpleNamespace(Generator=_generator))
    return made


def _ckpt_threads():
    return [t.name for t in threading.enumerate() if t.name.startswith("dfs_ckpt")]


def test_save_writes_total_bytes_in_32mib_chunks(monkeypatch):
    made = _fake_dgen(monkeypatch)
    client = _RecordingClient()
    ckpt = DfsStreamingCheckpointing(client=client)
    total = 2 * 32 * MiB + 5
    ckpt.save("checkpoints/global_epoch1_step1/model_states-0.pt", total)

    opens = [e for e in client.log if e[0] == "open_write"]
    writes = [e for e in client.log if e[0] == "write"]
    assert opens == [("open_write", "/checkpoints/global_epoch1_step1/model_states-0.pt")]
    assert [(off, n) for _, off, n, _ in writes] == [
        (0, 32 * MiB), (32 * MiB, 32 * MiB), (64 * MiB, 5)]
    assert client.log[-1] == ("close",)
    # One generator per file, built like StreamingCheckpointing's.
    assert len(made) == 1
    kwargs = made[0].kwargs
    assert kwargs["size"] == total
    assert kwargs["chunk_size"] == 32 * MiB
    assert kwargs["dedup_ratio"] == 1.0
    assert kwargs["compress_ratio"] == 1.0
    assert kwargs["numa_mode"] == "auto"
    assert kwargs["max_threads"] >= 1
    assert made[0].calls == 3


@pytest.mark.parametrize("total", [MiB - 1, MiB, MiB + 1, 3 * MiB, 5 * MiB + 77])
def test_save_file_has_exact_size(tmp_path, total):
    pytest.importorskip("dgen_py")
    client = DirBackedDfsClient(str(tmp_path))
    client.connect()
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    ckpt.save("/ckpt/size.pt", total)
    assert os.path.getsize(tmp_path / "ckpt" / "size.pt") == total


def test_save_payload_is_fresh_and_incompressible(tmp_path):
    pytest.importorskip("dgen_py")
    client = DirBackedDfsClient(str(tmp_path))
    client.connect()
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, num_buffers=4, client=client)
    nchunks = 6
    ckpt.save("/ckpt/payload.pt", nchunks * MiB)
    data = (tmp_path / "ckpt" / "payload.pt").read_bytes()
    chunks = [data[i * MiB:(i + 1) * MiB] for i in range(nchunks)]
    # The first 8 bytes are left out: a payload that only changes a stamp
    # there must not pass.
    body = [c[8:] for c in chunks]
    assert body[0] != body[1]
    assert body[1] != body[2]
    # The same buffer comes round again after num_buffers chunks.
    assert body[0] != body[4]
    assert body[1] != body[5]
    for chunk in chunks:
        assert len(zlib.compress(chunk)) >= 0.95 * len(chunk)


def test_save_requires_dgen(monkeypatch):
    monkeypatch.setattr(ckpt_mod, "_dgen_py", None)
    client = _RecordingClient()
    with pytest.raises(ImportError, match="dgen-py"):
        DfsStreamingCheckpointing(client=client).save("/c/x.pt", MiB)
    assert client.log == []


def test_save_producer_error_propagates_and_stops_threads(monkeypatch):
    _fake_dgen(monkeypatch, fail_at=3)
    before = _ckpt_threads()
    client = _RecordingClient()
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    with pytest.raises(ValueError, match="injected fill failure"):
        ckpt.save("/c/x.pt", 8 * MiB)
    assert _ckpt_threads() == before
    assert client.log[-1] == ("close",)
    assert len([e for e in client.log if e[0] == "write"]) == 2


def test_save_short_fill_is_an_error(monkeypatch):
    _fake_dgen(monkeypatch, short=True)
    before = _ckpt_threads()
    client = _RecordingClient()
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    with pytest.raises(RuntimeError, match="dgen-py filled"):
        ckpt.save("/c/x.pt", 2 * MiB)
    assert _ckpt_threads() == before
    assert client.log[-1] == ("close",)


def test_save_write_error_propagates_and_stops_threads(monkeypatch):
    _fake_dgen(monkeypatch)
    before = _ckpt_threads()
    client = _RecordingClient(fail_write_at=2)
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    with pytest.raises(OSError, match="injected write failure"):
        ckpt.save("/c/x.pt", 16 * MiB)
    assert _ckpt_threads() == before
    assert client.log[-1] == ("close",)


def test_save_zero_bytes_does_nothing():
    client = _RecordingClient()
    DfsStreamingCheckpointing(client=client).save("/c/x.pt", 0)
    assert client.log == []


def _reads(client):
    return sorted((off, n) for name, off, n in (e for e in client.log if e[0] == "read"))


def test_load_splits_file_into_reader_blocks():
    client = _RecordingClient(size=64 * MiB)
    ckpt = DfsStreamingCheckpointing(client=client)
    ckpt.load("/checkpoints/x.pt", 64 * MiB)
    assert _reads(client) == [(0, 16 * MiB), (16 * MiB, 16 * MiB),
                              (32 * MiB, 16 * MiB), (48 * MiB, 16 * MiB)]
    assert len([e for e in client.log if e[0] == "open_read"]) == 4
    assert not [t for t in _ckpt_threads() if t.startswith("dfs_ckpt_read")]


def test_load_reads_blocks_in_read_chunk_pieces():
    total = 2 * 32 * MiB + 7
    client = _RecordingClient(size=total)
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, num_parallel_readers=2, client=client)
    assert ckpt.read_chunk_size == 4 * MiB
    ckpt.load("/checkpoints/x.pt", total)
    block = -(-total // 2)
    expected = []
    for start in (0, block):
        end = min(start + block, total)
        pos = start
        while pos < end:
            n = min(4 * MiB, end - pos)
            expected.append((pos, n))
            pos += n
    assert _reads(client) == sorted(expected)
    assert sum(n for _, n in _reads(client)) == total


@pytest.mark.parametrize("readers", [1, 2, 4])
def test_load_reads_exactly_total_bytes(readers):
    total = 3 * MiB + 5
    client = _RecordingClient(size=total)
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, num_parallel_readers=readers,
                                     client=client)
    ckpt.load("/checkpoints/x.pt", total)
    reads = _reads(client)
    assert sum(n for _, n in reads) == total
    assert len([e for e in client.log if e[0] == "open_read"]) == readers


@pytest.mark.parametrize("readers", [1, 4])
def test_load_short_file_raises(readers):
    client = _RecordingClient(size=3 * MiB)
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, num_parallel_readers=readers,
                                     client=client)
    with pytest.raises(OSError, match="ended early"):
        ckpt.load("/checkpoints/x.pt", 3 * MiB + 1)
    assert not [t for t in _ckpt_threads() if t.startswith("dfs_ckpt_read")]


def test_save_then_load_through_dir_backed_client(tmp_path):
    pytest.importorskip("dgen_py")
    client = DirBackedDfsClient(str(tmp_path))
    client.connect()
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    total = 2 * MiB + 123
    ckpt.save("/ckpt/step1/layer_0.pt", total)
    assert os.path.getsize(tmp_path / "ckpt" / "step1" / "layer_0.pt") == total
    ckpt.load("/ckpt/step1/layer_0.pt", total)
    with pytest.raises(OSError, match="ended early"):
        ckpt.load("/ckpt/step1/layer_0.pt", total + 1)


def test_checkpoint_never_uses_local_open(monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError(f"local open called: {a!r}")

    _fake_dgen(monkeypatch)
    client = _RecordingClient(size=3 * MiB)
    ckpt = DfsStreamingCheckpointing(chunk_size=MiB, client=client)
    monkeypatch.setattr(builtins, "open", _forbidden)
    monkeypatch.setattr(io, "open", _forbidden)
    monkeypatch.setattr(os, "open", _forbidden)
    monkeypatch.setattr(os, "makedirs", _forbidden)
    ckpt.save("ckpt/a.pt", 3 * MiB)
    ckpt.load("ckpt/a.pt", 3 * MiB)


def test_default_client_resolves_checkpoint_inside_container(monkeypatch, tmp_path):
    from dlio_benchmark.utils.config import ConfigArguments

    _init_mpi()

    def _no_libdfs():
        raise AssertionError("libdfs must not be loaded in these tests")

    _fake_dgen(monkeypatch)
    args = ConfigArguments.get_instance()
    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dfs_client, "_registry", {})
    monkeypatch.setattr(dfs_client, "_load_lib", _no_libdfs)
    monkeypatch.setattr(args, "storage_type", StorageType.DAOS)
    monkeypatch.setattr(args, "storage_options", {"pool": "pool0", "container": "cont0"})
    try:
        ckpt = DfsStreamingCheckpointing(chunk_size=MiB)
        ckpt.save("checkpoints/global_epoch1_step1/model_states-0.pt", MiB + 1)
        local = tmp_path / "pool0" / "cont0" / "checkpoints" / "global_epoch1_step1" / "model_states-0.pt"
        assert local.stat().st_size == MiB + 1
    finally:
        dfs_client.close_all()

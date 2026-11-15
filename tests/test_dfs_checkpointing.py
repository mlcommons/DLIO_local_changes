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
import types
from types import SimpleNamespace

import pytest

from dlio_benchmark.checkpointing.dfs_streaming_checkpointing import DfsStreamingCheckpointing
from dlio_benchmark.common.enumerations import StorageType
from dlio_benchmark.storage import dfs_client
from dlio_benchmark.storage.dfs_testing import DirBackedDfsClient

MiB = 1024 * 1024


class _RecordingWriter:
    def __init__(self, log):
        self._log = log
        self._offset = 0
        self.closed = False

    def write(self, data):
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
    def __init__(self, size=0):
        self.log = []
        self.size = size

    def open_write(self, path):
        self.log.append(("open_write", path))
        return _RecordingWriter(self.log)

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
    assert PyTorchCheckpointing._get_streaming(daos_self) is streaming

    local_self = SimpleNamespace(args=SimpleNamespace(storage_type=StorageType.LOCAL_FS))
    assert isinstance(PyTorchCheckpointing._get_streaming(local_self), mlp_cls)


# -- save / load ---------------------------------------------------------------


def test_save_writes_total_bytes_in_32mib_chunks():
    client = _RecordingClient()
    ckpt = DfsStreamingCheckpointing(client=client)
    total = 2 * 32 * MiB + 5
    ckpt.save("checkpoints/global_epoch1_step1/model_states-0.pt", total)

    opens = [e for e in client.log if e[0] == "open_write"]
    writes = [e for e in client.log if e[0] == "write"]
    assert opens == [("open_write", "/checkpoints/global_epoch1_step1/model_states-0.pt")]
    assert [(off, n) for _, off, n, _ in writes] == [
        (0, 32 * MiB), (32 * MiB, 32 * MiB), (64 * MiB, 5)]
    assert len({head for _, _, _, head in writes}) == len(writes)
    assert client.log[-1] == ("close",)


def test_save_zero_bytes_does_nothing():
    client = _RecordingClient()
    DfsStreamingCheckpointing(client=client).save("/c/x.pt", 0)
    assert client.log == []


def test_load_reads_total_bytes():
    client = _RecordingClient(size=2 * 32 * MiB + 7)
    ckpt = DfsStreamingCheckpointing(client=client)
    ckpt.load("/checkpoints/x.pt", 2 * 32 * MiB + 7)
    reads = [e for e in client.log if e[0] == "read"]
    assert [(off, n) for _, off, n in reads] == [
        (0, 32 * MiB), (32 * MiB, 32 * MiB), (64 * MiB, 7)]


def test_save_then_load_through_dir_backed_client(tmp_path):
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

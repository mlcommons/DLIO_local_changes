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
Tests for ``storage_type: daos``: ``DfsStorage``, the pre-connect
configuration check, and the spawn policy.

Storage tests run against ``DirBackedDfsClient`` through
``DLIO_DAOS_TEST_BACKING_DIR``; libdfs is never loaded.
"""

import inspect
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dlio_benchmark.common.enumerations import (
    DataLoaderType, FormatType, FrameworkType, MetadataType, StorageType,
)
from dlio_benchmark.storage import dfs_client
from dlio_benchmark.storage.dfs_precheck import check_daos_config


def _no_libdfs():
    raise AssertionError("libdfs must not be loaded in these tests")


@pytest.fixture
def mpi_ready():
    """ConfigArguments() reads the MPI rank and size; make them available."""
    from dlio_benchmark.common.enumerations import MPIState
    from dlio_benchmark.utils.utility import DLIOMPI

    if DLIOMPI.get_instance().mpi_state == MPIState.CHILD_INITIALIZED:
        DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()


@pytest.fixture
def daos_args(mpi_ready, monkeypatch, tmp_path):
    from dlio_benchmark.utils.config import ConfigArguments

    args = ConfigArguments.get_instance()
    monkeypatch.setattr(args, "storage_type", StorageType.DAOS)
    monkeypatch.setattr(args, "storage_options", {"pool": "pool0", "container": "cont0"})
    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dfs_client, "_registry", {})
    monkeypatch.setattr(dfs_client, "_load_lib", _no_libdfs)
    yield args
    dfs_client.close_all()


@pytest.fixture
def container_dir(tmp_path):
    return tmp_path / "pool0" / "cont0"


def _storage(namespace="/"):
    from dlio_benchmark.storage.storage_factory import StorageFactory
    return StorageFactory().get_storage(StorageType.DAOS, namespace, None)


def test_factory_returns_dfs_storage(daos_args):
    from dlio_benchmark.storage.dfs_storage import DfsStorage
    assert isinstance(_storage(), DfsStorage)


def test_create_and_get_namespace(daos_args, container_dir):
    s = _storage("/bench/root")
    assert s.create_namespace(exist_ok=True) is True
    assert (container_dir / "bench" / "root").is_dir()
    assert s.get_namespace() == "/bench/root"
    assert s.get_uri("data/train/a.npz") == "/bench/root/data/train/a.npz"
    assert s.get_uri("/abs/path") == "/abs/path"

    rel = _storage("checkpoints/llama")
    assert rel.get_namespace() == "/checkpoints/llama"
    rel.create_namespace(exist_ok=True)
    assert (container_dir / "checkpoints" / "llama").is_dir()

    root = _storage("./")
    assert root.get_namespace() == "/"
    assert root.create_namespace(exist_ok=True) is True


def test_create_node_parents_exist_ok(daos_args, container_dir):
    s = _storage("/")
    s.create_node("data/train/", exist_ok=True)
    s.create_node("data/train/", exist_ok=True)
    assert (container_dir / "data" / "train").is_dir()
    with pytest.raises(FileExistsError):
        s.create_node("data/train", exist_ok=False)


def test_get_node_kinds(daos_args):
    s = _storage("/")
    s.create_node("d", exist_ok=True)
    s.put_data(s.get_uri("d/f.bin"), b"abc")
    assert s.get_node("d") == MetadataType.DIRECTORY
    assert s.get_node("d/f.bin") == MetadataType.FILE
    assert s.get_node("d/missing") is None
    assert s.get_node("d/f.bin/below") is None
    assert s.isfile(s.get_uri("d/f.bin")) is True
    assert s.isfile(s.get_uri("d")) is False
    assert s.file_exists(s.get_uri("d/f.bin")) is True
    assert s.file_exists(s.get_uri("d/nope")) is False


def test_walk_node_names(daos_args):
    s = _storage("/")
    s.create_node("data/train", exist_ok=True)
    names = [f"img_{i}_of_3.npz" for i in range(3)]
    for n in names:
        s.put_data(s.get_uri(f"data/train/{n}"), b"x")
    s.create_node("data/train/sub", exist_ok=True)
    assert sorted(s.walk_node("data/train")) == sorted(names + ["sub"])
    assert s.walk_node("data/missing") == []


def test_walk_node_pattern_returns_full_paths(daos_args):
    s = _storage("/")
    for sub in ("0", "1"):
        s.create_node(f"data/train/{sub}", exist_ok=True)
    s.put_data(s.get_uri("data/train/0/a.npz"), b"a")
    s.put_data(s.get_uri("data/train/1/b.npz"), b"b")
    s.put_data(s.get_uri("data/train/1/C.NPZ"), b"c")
    s.put_data(s.get_uri("data/train/1/d.txt"), b"d")
    s.put_data(s.get_uri("data/train/1/.hidden.npz"), b"h")

    one = s.walk_node("data/train/1/*.npz", use_pattern=True)
    assert sorted(one) == ["/data/train/1/C.NPZ", "/data/train/1/b.npz"]

    both = s.walk_node("data/train/*/*.npz", use_pattern=True)
    assert sorted(both) == [
        "/data/train/0/a.npz", "/data/train/1/C.NPZ", "/data/train/1/b.npz"]
    for path in both:
        assert s.get_data(path, None) in (b"a", b"b", b"c")

    assert s.walk_node("data/none/*.npz", use_pattern=True) == []
    with pytest.raises(Exception, match="Unknown file format"):
        s.walk_node("data/train/1/*.NPZ", use_pattern=True)


def test_put_get_roundtrip_and_range(daos_args, container_dir):
    s = _storage("/")
    s.create_node("data", exist_ok=True)
    payload = bytes(range(200))
    uri = s.get_uri("data/x.bin")
    s.put_data(uri, io.BytesIO(payload))
    assert (container_dir / "data" / "x.bin").read_bytes() == payload
    assert s.get_data(uri, None) == payload
    assert s.get_data(uri, None, offset=10, length=5) == payload[10:15]
    assert s.get_data(uri, None, offset=190) == payload[190:]
    s.put_data(uri, memoryview(b"short"))
    assert s.get_data(uri, None) == b"short"
    with pytest.raises(FileNotFoundError):
        s.get_data(s.get_uri("data/none.bin"), None)


def test_delete_node_recursive(daos_args, container_dir):
    s = _storage("/")
    s.create_node("data/train/0", exist_ok=True)
    s.put_data(s.get_uri("data/train/0/a.npz"), b"a")
    assert s.delete_node("data") is True
    assert not (container_dir / "data").exists()
    assert s.get_node("data") is None


def test_islocalfs_false(daos_args):
    s = _storage("/")
    assert s.islocalfs() is False
    assert s.get_basename("/a/b/c.npz") == "c.npz"


def test_storage_requires_pool_and_container(daos_args, monkeypatch):
    monkeypatch.setattr(daos_args, "storage_options", {"container": "cont0"})
    with pytest.raises(Exception, match="storage_options.pool"):
        _storage("/")


# -- Pre-connect configuration check -----------------------------------------


def _precheck_args(**overrides):
    base = dict(
        storage_type=StorageType.DAOS,
        storage_options={"pool": "pool0", "container": "cont0"},
        data_loader=DataLoaderType.PYTORCH,
        odirect=False,
        format=FormatType.NPZ,
        do_train=True,
        do_eval=False,
        generate_data=False,
        do_checkpoint=False,
        framework=FrameworkType.PYTORCH,
        parquet_partition_by=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_precheck_accepts_supported_config():
    for fmt in (FormatType.NPZ, FormatType.NPY, FormatType.JPEG, FormatType.PNG,
                FormatType.PARQUET):
        check_daos_config(_precheck_args(format=fmt))
    check_daos_config(_precheck_args(format=FormatType.TFRECORD,
                                     data_loader=DataLoaderType.TENSORFLOW,
                                     framework=FrameworkType.TENSORFLOW))


def test_precheck_ignores_other_storage_types():
    check_daos_config(_precheck_args(storage_type=StorageType.LOCAL_FS,
                                     storage_options=None, format=FormatType.HDF5,
                                     odirect=True))


def test_precheck_missing_pool():
    with pytest.raises(Exception, match=r"storage_options\.pool"):
        check_daos_config(_precheck_args(storage_options={"container": "cont0"}))
    with pytest.raises(Exception, match=r"storage_options\.pool"):
        check_daos_config(_precheck_args(storage_options={"pool": "", "container": "c"}))
    with pytest.raises(Exception, match=r"storage_options\.pool"):
        check_daos_config(_precheck_args(storage_options=None))


def test_precheck_missing_container():
    with pytest.raises(Exception, match=r"storage_options\.container"):
        check_daos_config(_precheck_args(storage_options={"pool": "pool0"}))


def test_precheck_rejects_odirect():
    with pytest.raises(Exception, match="odirect"):
        check_daos_config(_precheck_args(odirect=True))


def test_precheck_rejects_dali():
    with pytest.raises(Exception, match="native_dali"):
        check_daos_config(_precheck_args(data_loader=DataLoaderType.NATIVE_DALI,
                                         format=FormatType.NPY))


def test_precheck_rejects_hdf5():
    with pytest.raises(Exception, match="format=hdf5"):
        check_daos_config(_precheck_args(format=FormatType.HDF5))
    with pytest.raises(Exception, match="format=csv"):
        check_daos_config(_precheck_args(format=FormatType.CSV, do_train=False,
                                         generate_data=True))


def test_precheck_rejects_tfrecord_without_tensorflow_loader():
    with pytest.raises(Exception, match="tfrecord only with reader.data_loader=tensorflow"):
        check_daos_config(_precheck_args(format=FormatType.TFRECORD))
    check_daos_config(_precheck_args(format=FormatType.TFRECORD, do_train=False,
                                     generate_data=True))


def test_precheck_rejects_parquet_partition_by():
    with pytest.raises(Exception, match="partition_by"):
        check_daos_config(_precheck_args(format=FormatType.PARQUET, do_train=False,
                                         generate_data=True,
                                         parquet_partition_by="day"))


def test_precheck_rejects_non_pytorch_checkpoint():
    with pytest.raises(Exception, match="checkpointing requires framework=pytorch"):
        check_daos_config(_precheck_args(do_train=False, do_checkpoint=True,
                                         framework=FrameworkType.TENSORFLOW))


def test_precheck_runs_before_storage_creation(mpi_ready, monkeypatch):
    from dlio_benchmark import main as dlio_main

    def _fake_load(args, cfg):
        monkeypatch.setattr(args, "storage_type", StorageType.DAOS)
        monkeypatch.setattr(args, "storage_options", {"container": "cont0"})

    class _NoStorageFactory:
        def get_storage(self, *a, **k):
            raise AssertionError("storage was created before the DAOS check")

    def _no_client(*a, **k):
        raise AssertionError("get_client was called before the DAOS check")

    monkeypatch.setattr(dlio_main, "LoadConfig", _fake_load)
    monkeypatch.setattr(dlio_main, "StorageFactory", _NoStorageFactory)
    monkeypatch.setattr(dfs_client, "get_client", _no_client)
    with pytest.raises(Exception, match=r"storage_options\.pool"):
        dlio_main.DLIOBenchmark({})


# -- Spawn policy ------------------------------------------------------------


def test_daos_promotes_fork_to_spawn():
    from dlio_benchmark.utils import config as config_mod
    from dlio_benchmark.utils.config import ConfigArguments

    args = SimpleNamespace(storage_type=StorageType.DAOS,
                           multiprocessing_context="fork", logger=MagicMock())
    ConfigArguments._apply_daos_spawn_policy(args)
    assert args.multiprocessing_context == "spawn"

    explicit = SimpleNamespace(storage_type=StorageType.DAOS,
                               multiprocessing_context="forkserver", logger=MagicMock())
    ConfigArguments._apply_daos_spawn_policy(explicit)
    assert explicit.multiprocessing_context == "forkserver"

    local = SimpleNamespace(storage_type=StorageType.LOCAL_FS,
                            multiprocessing_context="fork", logger=MagicMock())
    ConfigArguments._apply_daos_spawn_policy(local)
    assert local.multiprocessing_context == "fork"

    assert "self._apply_daos_spawn_policy()" in inspect.getsource(config_mod)

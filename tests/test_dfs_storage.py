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
    monkeypatch.setattr(args, "daos_test_stand_in", True)
    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, "1")
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


def test_precheck_sets_stand_in_flag_from_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV, raising=False)
    monkeypatch.delenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, raising=False)
    args = _precheck_args()
    check_daos_config(args)
    assert args.daos_test_stand_in is False

    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, "1")
    args = _precheck_args()
    check_daos_config(args)
    assert args.daos_test_stand_in is True

    monkeypatch.delenv(dfs_client.TEST_ALLOW_STAND_IN_ENV)
    with pytest.raises(RuntimeError, match=dfs_client.TEST_ALLOW_STAND_IN_ENV):
        check_daos_config(_precheck_args())


class _FakeMPI:
    """A DLIOMPI stand-in: each reduction adds what the other ranks
    contribute; ``other_ranks=None`` forbids any reduction."""

    def __init__(self, mpi_state, size, other_ranks=None):
        self.mpi_state = mpi_state
        self._size = size
        self._other_ranks = other_ranks
        self.sent = []

    def size(self):
        return self._size

    def reduce(self, num):
        assert self._other_ranks is not None, "no reduction expected"
        self.sent.append(num)
        return num + self._other_ranks[len(self.sent) - 1]


def _use_fake_mpi(monkeypatch, other_ranks, mpi_state=None, size=2):
    """Install a ``_FakeMPI``; ``other_ranks`` is ``[selected, invalid]``
    contributed by the other rank of a two-rank run."""
    from dlio_benchmark.common.enumerations import MPIState
    from dlio_benchmark.utils.utility import DLIOMPI

    if mpi_state is None:
        mpi_state = MPIState.MPI_INITIALIZED
    fake = _FakeMPI(mpi_state, size, other_ranks)
    monkeypatch.setattr(DLIOMPI, "get_instance", staticmethod(lambda: fake))
    return fake


def _stand_in_env(monkeypatch, tmp_path, backing, allow):
    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV, raising=False)
    monkeypatch.delenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, raising=False)
    if backing:
        monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    if allow:
        monkeypatch.setenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, "1")


def test_stand_in_invalid_rank_stops_valid_rank(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    _stand_in_env(monkeypatch, tmp_path, backing=False, allow=False)
    fake = _use_fake_mpi(monkeypatch, [0, 1])
    args = _precheck_args()
    with pytest.raises(RuntimeError, match="1 of 2 MPI ranks"):
        resolve_daos_stand_in(args)
    assert fake.sent == [0, 0]
    assert not hasattr(args, "daos_test_stand_in")


def test_stand_in_invalid_rank_joins_both_reductions(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    _stand_in_env(monkeypatch, tmp_path, backing=True, allow=False)
    fake = _use_fake_mpi(monkeypatch, [0, 0])
    args = _precheck_args()
    with pytest.raises(RuntimeError, match=dfs_client.TEST_ALLOW_STAND_IN_ENV) as info:
        resolve_daos_stand_in(args)
    assert "MPI ranks" not in str(info.value)
    assert fake.sent == [0, 1]
    assert not hasattr(args, "daos_test_stand_in")


def test_stand_in_mixed_ranks_stop_rank_without_it(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    _stand_in_env(monkeypatch, tmp_path, backing=False, allow=False)
    fake = _use_fake_mpi(monkeypatch, [1, 0])
    args = _precheck_args()
    with pytest.raises(RuntimeError, match="does not select"):
        resolve_daos_stand_in(args)
    assert fake.sent == [0, 0]
    assert not hasattr(args, "daos_test_stand_in")


def test_stand_in_mixed_ranks_stop_rank_with_it(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    _stand_in_env(monkeypatch, tmp_path, backing=True, allow=True)
    fake = _use_fake_mpi(monkeypatch, [0, 0])
    args = _precheck_args()
    with pytest.raises(RuntimeError, match="this rank selects"):
        resolve_daos_stand_in(args)
    assert fake.sent == [1, 0]
    assert not hasattr(args, "daos_test_stand_in")


def test_stand_in_flag_agreed_across_ranks(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    _stand_in_env(monkeypatch, tmp_path, backing=True, allow=True)
    fake = _use_fake_mpi(monkeypatch, [1, 0])
    args = _precheck_args()
    resolve_daos_stand_in(args)
    assert args.daos_test_stand_in is True
    assert fake.sent == [1, 0]

    _stand_in_env(monkeypatch, tmp_path, backing=False, allow=False)
    fake = _use_fake_mpi(monkeypatch, [0, 0])
    args = _precheck_args()
    resolve_daos_stand_in(args)
    assert args.daos_test_stand_in is False
    assert fake.sent == [0, 0]


@pytest.mark.parametrize("state_name, size", [
    ("UNINITIALIZED", 4),
    ("CHILD_INITIALIZED", 4),
    ("MPI_INITIALIZED", 1),
])
def test_stand_in_no_reduction_outside_multi_rank_mpi(monkeypatch, tmp_path,
                                                       state_name, size):
    from dlio_benchmark.common.enumerations import MPIState
    from dlio_benchmark.storage.dfs_precheck import resolve_daos_stand_in

    state = getattr(MPIState, state_name)

    _stand_in_env(monkeypatch, tmp_path, backing=True, allow=True)
    fake = _use_fake_mpi(monkeypatch, None, mpi_state=state, size=size)
    args = _precheck_args()
    resolve_daos_stand_in(args)
    assert args.daos_test_stand_in is True
    assert fake.sent == []

    _stand_in_env(monkeypatch, tmp_path, backing=False, allow=False)
    fake = _use_fake_mpi(monkeypatch, None, mpi_state=state, size=size)
    args = _precheck_args()
    resolve_daos_stand_in(args)
    assert args.daos_test_stand_in is False
    assert fake.sent == []


def test_connection_options_reject_stand_in_mismatch(monkeypatch, tmp_path):
    from dlio_benchmark.storage.dfs_precheck import daos_connection_options

    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(dfs_client.TEST_ALLOW_STAND_IN_ENV, "1")
    assert daos_connection_options(_precheck_args(daos_test_stand_in=True)) == \
        ("pool0", "cont0", None)
    with pytest.raises(RuntimeError, match="daos_test_stand_in"):
        daos_connection_options(_precheck_args(daos_test_stand_in=False))
    with pytest.raises(RuntimeError, match="daos_test_stand_in"):
        daos_connection_options(_precheck_args())

    monkeypatch.delenv(dfs_client.TEST_BACKING_DIR_ENV)
    with pytest.raises(RuntimeError, match="daos_test_stand_in"):
        daos_connection_options(_precheck_args(daos_test_stand_in=True))
    assert daos_connection_options(_precheck_args(daos_test_stand_in=False)) == \
        ("pool0", "cont0", None)


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


# -- dlio_postprocessor ------------------------------------------------------


def _postprocessor_args(output_folder):
    from collections import namedtuple

    args = {
        'output_folder': str(output_folder),
        'name': '',
        'num_proc': 2,
        'epochs': 2,
        'do_eval': False,
        'do_checkpoint': False,
        'batch_size': 4,
        'batch_size_eval': 1,
        'record_size': 234560851,
    }
    return namedtuple('args', args.keys())(*args.values())


def _copy_postprocessor_data(tmp_path, mark=None):
    import json
    import shutil
    from pathlib import Path

    folder = tmp_path / "output"
    shutil.copytree(Path(__file__).resolve().parent / "test_data", folder)
    if mark is not None:
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["daos_test_stand_in"] = mark
        summary_path.write_text(json.dumps(summary))
    return folder


def test_postprocessor_refuses_stand_in_summary(tmp_path):
    postprocessor = pytest.importorskip("dlio_benchmark.postprocessor")

    folder = _copy_postprocessor_data(tmp_path, mark=True)
    with pytest.raises(RuntimeError, match="daos_test_stand_in"):
        postprocessor.DLIOPostProcessor(_postprocessor_args(folder))
    assert list(folder.glob("DLIO*report.txt")) == []


def test_postprocessor_accepts_unmarked_summary(tmp_path):
    postprocessor = pytest.importorskip("dlio_benchmark.postprocessor")

    folder = _copy_postprocessor_data(tmp_path)
    postproc = postprocessor.DLIOPostProcessor(_postprocessor_args(folder))
    postproc.process_loading_and_processing_times()
    assert postproc.overall_stats['samples/s']['mean'] == '5.10'

    folder = _copy_postprocessor_data(tmp_path / "false_mark", mark=False)
    postprocessor.DLIOPostProcessor(_postprocessor_args(folder))

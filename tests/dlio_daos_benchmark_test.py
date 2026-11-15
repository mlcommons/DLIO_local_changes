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
End-to-end DLIOBenchmark runs with ``storage_type: daos``.

The DFS client is replaced by the directory-backed stand-in selected with
``DLIO_DAOS_TEST_BACKING_DIR``; the variable is inherited by spawned
DataLoader workers, so the spawn cases see the same container contents.
libdfs is never loaded.  Each run executes in a scratch working directory
so that any accidental local write would be detected.

Run without mpirun (single rank), like tests/test_fast_ci.py.
"""

import os

import pytest
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

import dlio_benchmark
from dlio_benchmark.storage import dfs_client
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from tests.utils import TEST_TIMEOUT_SECONDS

CONFIG_DIR = os.path.dirname(dlio_benchmark.__file__) + "/configs/"
DLIO_TEST_OUTPUT_DIR = os.environ.get("DLIO_TEST_OUTPUT_DIR", "dlio_test_output")
POOL = "pool0"
CONT = "cont0"
NUM_FILES_TRAIN = 8
NUM_FILES_EVAL = 2


def _no_libdfs():
    raise AssertionError("libdfs must not be loaded in these tests")


def _reset_singletons():
    # framework.py calls DLIOMPI.comm() at import time, so MPI must be
    # initialized before the framework modules below are first imported.
    DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()
    ConfigArguments.reset()
    try:
        from dlio_benchmark.checkpointing.pytorch_checkpointing import PyTorchCheckpointing
        from dlio_benchmark.framework.torch_framework import TorchFramework
        PyTorchCheckpointing._PyTorchCheckpointing__instance = None
        TorchFramework._TorchFramework__instance = None
    except ImportError:
        pass
    try:
        from dlio_benchmark.framework.tf_framework import TFFramework
        TFFramework._TFFramework__instance = None
    except ImportError:
        pass


@pytest.fixture
def daos_env(tmp_path, monkeypatch):
    backing = tmp_path / "backing"
    workdir = tmp_path / "cwd"
    backing.mkdir()
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(backing))
    monkeypatch.setattr(dfs_client, "_registry", {})
    monkeypatch.setattr(dfs_client, "_load_lib", _no_libdfs)
    _reset_singletons()
    yield backing / POOL / CONT, workdir
    dfs_client.close_all()
    _reset_singletons()
    DLIOMPI.reset()


def _overrides(storage_root="/", epochs=1):
    return [
        "++workload.storage.storage_type=daos",
        f"++workload.storage.storage_root={storage_root}",
        f"++workload.storage.storage_options.pool={POOL}",
        f"++workload.storage.storage_options.container={CONT}",
        "++workload.storage.storage_options.stagger_workers=false",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
        f"++workload.dataset.num_files_train={NUM_FILES_TRAIN}",
        f"++workload.dataset.num_files_eval={NUM_FILES_EVAL}",
        f"++workload.train.epochs={epochs}",
        "++workload.train.computation_time=0.01",
        "++workload.evaluation.eval_time=0.005",
    ]


def _run(overrides):
    DLIOMPI.get_instance().initialize()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    ConfigArguments.reset()
    workload = OmegaConf.to_container(cfg["workload"], resolve=True)
    workload.setdefault("output", {})["folder"] = DLIO_TEST_OUTPUT_DIR
    from dlio_benchmark.main import DLIOBenchmark
    benchmark = DLIOBenchmark(workload)
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    return benchmark


def _files(container_dir, rel, ext):
    folder = container_dir / rel
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.name.endswith("." + ext))


_DATA_SUFFIXES = (".npz", ".npy", ".jpeg", ".png", ".tfrecord", ".parquet", ".pt")


def _assert_no_local_data(workdir):
    leaked = [p for p in workdir.rglob("*")
              if {"dlio_test", "dlio_root", "checkpoints"} & set(p.relative_to(workdir).parts)
              or p.suffix in _DATA_SUFFIXES]
    assert leaked == [], f"local writes outside the container: {leaked}"


def _train_steps_per_epoch(benchmark):
    """Completed training steps of each epoch, as counted by the stats."""
    steps = {}
    for epoch, out in benchmark.stats.output.items():
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            continue
        steps[epoch] = sum(len(v) for k, v in out["proc"].items() if k.startswith("block"))
    return steps


def _generate_list_train_delete(container_dir, workdir, fmt, framework, read_threads, context,
                                extra=(), epochs=1):
    data_folder = f"/dlio_test/{fmt}"
    base = _overrides(epochs=epochs) + [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={framework}",
        f"++workload.dataset.format={fmt}",
        f"++workload.dataset.data_folder={data_folder}",
        f"++workload.reader.read_threads={read_threads}",
    ] + list(extra)
    if context is not None:
        base.append(f"++workload.reader.multiprocessing_context={context}")

    # generate
    _run(base + ["++workload.workflow.generate_data=True", "++workload.workflow.train=False"])
    _reset_singletons()

    # list
    train_files = _files(container_dir, f"dlio_test/{fmt}/train", fmt)
    valid_files = _files(container_dir, f"dlio_test/{fmt}/valid", fmt)
    assert len(train_files) == NUM_FILES_TRAIN
    assert len(valid_files) == NUM_FILES_EVAL

    # train
    benchmark = _run(base + ["++workload.workflow.generate_data=False",
                             "++workload.workflow.train=True"])
    args = ConfigArguments.get_instance()
    if read_threads > 0 and framework == "pytorch":
        assert args.multiprocessing_context == "spawn"
    outputs = [p for p in (workdir / DLIO_TEST_OUTPUT_DIR).glob("*_output.json")]
    assert len(outputs) >= 1
    if epochs > 1:
        expected = benchmark.steps_per_epoch
        assert expected > 0
        assert _train_steps_per_epoch(benchmark) == {e: expected for e in range(1, epochs + 1)}

    # delete
    benchmark.storage.delete_node(data_folder)
    assert benchmark.storage.get_node(data_folder) is None
    assert not (container_dir / "dlio_test" / fmt).exists()
    _assert_no_local_data(workdir)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("fmt, read_threads, context", [
    ("npz", 0, None),
    ("npy", 0, None),
    ("jpeg", 0, None),
    ("png", 0, None),
    ("npz", 2, "spawn"),
    ("npy", 2, "fork"),
    ("png", 2, "spawn"),
])
def test_daos_pytorch_formats(daos_env, fmt, read_threads, context):
    container_dir, workdir = daos_env
    _generate_list_train_delete(container_dir, workdir, fmt, "pytorch", read_threads, context)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
@pytest.mark.parametrize("read_threads, context", [(0, None), (2, "spawn")])
def test_daos_pytorch_parquet(daos_env, read_threads, context):
    pytest.importorskip("pyarrow")
    container_dir, workdir = daos_env
    _generate_list_train_delete(container_dir, workdir, "parquet", "pytorch", read_threads, context,
                                extra=["++workload.dataset.num_samples_per_file=4"])


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_daos_tensorflow_tfrecord(daos_env):
    pytest.importorskip("tensorflow")
    container_dir, workdir = daos_env
    # Two epochs: the reader's result is iterated by the pre-warm and by
    # every epoch, and each pass must cover the full file list.
    _generate_list_train_delete(container_dir, workdir, "tfrecord", "tensorflow", 2, None,
                                epochs=2)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_daos_pytorch_checkpoint_write_read(daos_env):
    pytest.importorskip("torch")
    container_dir, workdir = daos_env
    # A storage_root other than "/" checks that checkpoint_folder is resolved
    # from the container root both when writing and when counting checkpoints.
    overrides = _overrides(storage_root="/dlio_root") + [
        "++workload.framework=pytorch",
        "++workload.reader.data_loader=pytorch",
        "++workload.workflow.generate_data=False",
        "++workload.workflow.train=False",
        "++workload.workflow.checkpoint=True",
        "++workload.checkpoint.checkpoint_folder=dlio_test/checkpoints",
        "++workload.checkpoint.num_checkpoints_write=2",
        "++workload.checkpoint.num_checkpoints_read=1",
        "++workload.checkpoint.time_between_checkpoints=0",
        "++workload.checkpoint.randomize_tensor=False",
        "++workload.model.model_size_bytes=4096",
        "++workload.model.num_layers=2",
        "++workload.model.layer_parameters=[16]",
        "++workload.model.optimization_groups=[1024,128]",
    ]
    _run(overrides)
    ckpt_dir = container_dir / "dlio_test" / "checkpoints"
    steps = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
    assert steps == ["global_epoch1_step1", "global_epoch1_step2"]
    for step in steps:
        files = [p for p in (ckpt_dir / step).iterdir() if p.is_file()]
        assert files, f"no checkpoint files in {step}"
        assert all(p.suffix == ".pt" and p.stat().st_size > 0 for p in files)
    _assert_no_local_data(workdir)


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS, method="thread")
def test_daos_rejects_missing_container_before_connecting(daos_env):
    _, workdir = daos_env
    overrides = [o for o in _overrides() if "storage_options.container" not in o] + [
        "++workload.workflow.generate_data=True",
        "++workload.workflow.train=False",
    ]
    DLIOMPI.get_instance().initialize()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    ConfigArguments.reset()
    from dlio_benchmark.main import DLIOBenchmark
    with pytest.raises(Exception, match=r"storage_options\.container"):
        DLIOBenchmark(OmegaConf.to_container(cfg["workload"], resolve=True))
    assert dfs_client._registry == {}

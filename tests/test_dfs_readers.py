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
Tests for the ``storage_type: daos`` readers, their selection by the reader
factory and the PyTorch DataLoader, and the worker connection lifetime.

Data lives in ``DirBackedDfsClient`` (``DLIO_DAOS_TEST_BACKING_DIR``), so
spawned processes see the same files and libdfs is never loaded.

Top-level imports stay light on purpose: the worker-lifetime tests run
``_worker_child`` in a spawned process, which imports this module.
"""

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from dlio_benchmark.common.enumerations import (
    DataLoaderType, DatasetType, FormatType, StorageType,
)
from dlio_benchmark.storage import dfs_client

POOL = "pool0"
CONT = "cont0"


def _no_libdfs():
    raise AssertionError("libdfs must not be loaded in these tests")


@pytest.fixture
def reader_args(monkeypatch, tmp_path):
    from dlio_benchmark.common.enumerations import MPIState
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI

    if DLIOMPI.get_instance().mpi_state == MPIState.CHILD_INITIALIZED:
        DLIOMPI.reset()
    DLIOMPI.get_instance().initialize()
    args = ConfigArguments.get_instance()
    monkeypatch.setenv(dfs_client.TEST_BACKING_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(dfs_client, "_registry", {})
    monkeypatch.setattr(dfs_client, "_load_lib", _no_libdfs)
    monkeypatch.setattr(args, "storage_type", StorageType.DAOS)
    monkeypatch.setattr(args, "storage_options",
                        {"pool": POOL, "container": CONT, "stagger_workers": "false"})
    monkeypatch.setattr(args, "data_loader", DataLoaderType.PYTORCH)
    monkeypatch.setattr(args, "odirect", False)
    monkeypatch.setattr(args, "reader_class", None)
    monkeypatch.setattr(args, "batch_size", 2)
    monkeypatch.setattr(args, "num_samples_per_file", 1)
    monkeypatch.setattr(args, "resized_image", np.zeros(4, dtype=np.uint8), raising=False)
    monkeypatch.setattr(args, "record_length", args.record_length)
    monkeypatch.setattr(args, "train_file_map", {0: []}, raising=False)
    monkeypatch.setattr(args, "train_global_index_map", {}, raising=False)
    monkeypatch.setattr(args, "val_file_map", {0: []}, raising=False)
    monkeypatch.setattr(args, "val_global_index_map", {}, raising=False)
    monkeypatch.setattr(args, "file_map", [], raising=False)
    monkeypatch.setattr(args, "logfile_path", str(tmp_path / "dlio.log"),
                        raising=False)
    root = logging.getLogger()
    saved_level = root.level
    yield args
    dfs_client.close_all()
    # worker_init(-1) unpickles args and installs a new singleton, which
    # monkeypatch does not restore; drop whichever instance is current.
    ConfigArguments.reset()
    for handler in root.handlers[:]:
        if (isinstance(handler, logging.FileHandler)
                and handler.baseFilename.startswith(str(tmp_path))):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def _write_files(sizes, folder="/data/train", ext="npz"):
    client = dfs_client.get_client(POOL, CONT)
    client.mkdir(folder)
    paths = []
    for i, size in enumerate(sizes):
        path = f"{folder}/img_{i}_of_{len(sizes)}.{ext}"
        client.write(path, os.urandom(size))
        paths.append(path)
    return paths


def _make_reader(fmt):
    from dlio_benchmark.reader.reader_factory import ReaderFactory
    return ReaderFactory.get_reader(type=fmt, dataset_type=DatasetType.TRAIN,
                                    thread_index=0, epoch_number=1)


def _install(reader, paths):
    reader.file_map[reader.thread_index] = [(i, p, 0) for i, p in enumerate(paths)]


# -- Selection -----------------------------------------------------------------


def _assert_factory_selects(reader_args, monkeypatch, fmt, cls_name):
    monkeypatch.setattr(reader_args, "format", fmt)
    reader = _make_reader(fmt)
    assert type(reader).__name__ == cls_name
    from dlio_benchmark.reader._dfs_iterable_mixin import _DfsIterableMixin
    from dlio_benchmark.reader.reader_handler import FormatReader
    assert isinstance(reader, _DfsIterableMixin)
    assert type(reader).__bases__ == (FormatReader, _DfsIterableMixin)


def test_factory_selects_dfs_reader_npz(reader_args, monkeypatch):
    _assert_factory_selects(reader_args, monkeypatch, FormatType.NPZ, "NPZReaderDfsIterable")


def test_factory_selects_dfs_reader_npy(reader_args, monkeypatch):
    _assert_factory_selects(reader_args, monkeypatch, FormatType.NPY, "NPYReaderDfsIterable")


def test_factory_selects_dfs_reader_jpeg(reader_args, monkeypatch):
    _assert_factory_selects(reader_args, monkeypatch, FormatType.JPEG, "ImageReaderDfsIterable")


def test_factory_selects_dfs_reader_png(reader_args, monkeypatch):
    _assert_factory_selects(reader_args, monkeypatch, FormatType.PNG, "ImageReaderDfsIterable")


def test_factory_rejects_unsupported_format_for_daos(reader_args, monkeypatch):
    monkeypatch.setattr(reader_args, "format", FormatType.HDF5)
    with pytest.raises(Exception, match="storage_type=daos"):
        _make_reader(FormatType.HDF5)


def test_torch_loader_selects_iterable_dataset_for_daos(reader_args, monkeypatch):
    from dlio_benchmark.data_loader.torch_data_loader import (
        TorchDataLoader, TorchIterableDatasetSimple,
    )
    monkeypatch.setattr(reader_args, "format", FormatType.NPZ)
    monkeypatch.setattr(reader_args, "read_threads", 1)
    monkeypatch.setattr(reader_args, "multiprocessing_context", "spawn")
    loader = TorchDataLoader(FormatType.NPZ, DatasetType.TRAIN, 1)
    loader.read()
    assert isinstance(loader._dataset.dataset, TorchIterableDatasetSimple)
    assert loader._dataset.multiprocessing_context is not None


# -- Prefetch ------------------------------------------------------------------


def test_prefetch_records_byte_counts(reader_args, monkeypatch):
    sizes = [10, 2000, 0, 77, 4096, 1]
    paths = _write_files(sizes)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)

    seen = {}
    original = reader.get_sample

    def _record(filename, sample_index):
        seen[filename] = reader.open(filename)
        return original(filename, sample_index)

    monkeypatch.setattr(reader, "get_sample", _record)
    batches = list(reader.next())
    assert len(batches) == len(sizes) // reader_args.batch_size
    assert all(b is reader_args.resized_image for b in batches)
    assert seen == dict(zip(paths, sizes))
    assert reader._total_bytes_read == sum(sizes)
    assert reader._total_objects_read == len(sizes)
    reader.finalize_dfs_bytes()
    assert reader_args.record_length == sum(sizes) // len(sizes)
    assert reader._total_bytes_read == 0


def test_read_index_reads_through_dfs(reader_args, monkeypatch):
    paths = _write_files([123, 456])
    reader = _make_reader(FormatType.NPY)
    monkeypatch.setitem(reader.global_index_map, 0, (paths[1], 0))
    result = reader.read_index(0, 1)
    assert result is reader_args.resized_image
    assert reader._total_bytes_read == 456


def _read_order(reader, monkeypatch):
    """Record the peak number of reads started but not yet yielded."""
    import threading

    state = {"started": [], "outstanding": 0, "peak": 0}
    lock = threading.Lock()
    original_read = reader._dfs_read_one
    original_account = reader._dfs_account

    def _read(path):
        with lock:
            state["started"].append(path)
            state["outstanding"] += 1
            state["peak"] = max(state["peak"], state["outstanding"])
        return original_read(path)

    def _account(nbytes):
        with lock:
            state["outstanding"] -= 1
        return original_account(nbytes)

    monkeypatch.setattr(reader, "_dfs_read_one", _read)
    monkeypatch.setattr(reader, "_dfs_account", _account)
    return state


def _record_submits(reader, monkeypatch, on_submit=None):
    """Record the paths submitted to the reader's read pool, in order.

    ``on_submit(path, future)`` may adjust the future before it is queued.
    """
    submitted = []
    pool = reader._dfs_pool()
    real_submit = pool.submit

    def _submit(fn, path, *a, **k):
        future = real_submit(fn, path, *a, **k)
        submitted.append(path)
        if on_submit is not None:
            on_submit(path, future)
        return future

    monkeypatch.setattr(pool, "submit", _submit)
    return submitted


def test_default_prefetch_window_is_4(reader_args):
    from dlio_benchmark.reader import _dfs_iterable_mixin as mixin_mod

    assert mixin_mod._DEFAULT_PREFETCH_WINDOW == 4
    reader = _make_reader(FormatType.NPZ)
    assert reader._dfs_window_size() == 4


@pytest.mark.parametrize("window", [1, 4, 9])
def test_readahead_keeps_order_and_counts_once(reader_args, monkeypatch, window):
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", window)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    sizes = [10, 20, 30, 40, 50, 60]
    paths = _write_files(sizes)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    submitted = _record_submits(reader, monkeypatch)
    consumed = []
    original = reader.get_sample

    def _record(filename, sample_index):
        consumed.append((filename, reader.open(filename)))
        return original(filename, sample_index)

    monkeypatch.setattr(reader, "get_sample", _record)
    assert len(list(reader.next())) == len(paths)
    assert consumed == list(zip(paths, sizes))
    assert submitted == paths
    assert reader._total_bytes_read == sum(sizes)
    assert reader._total_objects_read == len(paths)


@pytest.mark.parametrize("window", [1, 2, 4, 12])
def test_readahead_outstanding_is_window_while_consuming(reader_args, monkeypatch, window):
    """Outstanding = submitted but not yet yielded, excluding the file being
    consumed.  It is ``window`` (or every remaining file) at each yield."""
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", window)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    paths = _write_files([16] * 10)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    state = _read_order(reader, monkeypatch)
    submitted = _record_submits(reader, monkeypatch)
    consumed = 0
    for _ in reader.next():
        consumed += 1
        assert len(submitted) - consumed == min(window, len(paths) - consumed)
    assert consumed == len(paths)
    assert submitted == paths
    assert 1 <= state["peak"] <= window


def test_window_1_submits_next_read_before_consumer_resumes(reader_args, monkeypatch):
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 1)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    paths = _write_files([8] * 3)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    submitted = _record_submits(reader, monkeypatch)
    gen = reader.next()
    next(gen)
    assert submitted == paths[0:2]
    next(gen)
    assert submitted == paths
    next(gen)
    assert submitted == paths
    with pytest.raises(StopIteration):
        next(gen)


def test_read_pool_is_reused_and_capped(reader_args, monkeypatch):
    from dlio_benchmark.reader import _dfs_iterable_mixin as mixin_mod

    assert mixin_mod._MAX_PREFETCH_CONCURRENCY == 64
    requested = []
    real_executor = mixin_mod.ThreadPoolExecutor

    class _Recording(real_executor):
        def __init__(self, max_workers=None, *a, **k):
            requested.append(max_workers)
            super().__init__(max_workers, *a, **k)

    monkeypatch.setattr(mixin_mod, "ThreadPoolExecutor", _Recording)
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 100)
    paths = _write_files([1] * 100, folder="/data/many")
    reader = _make_reader(FormatType.PNG)
    assert reader._dfs_read_pool is None
    _install(reader, paths)
    assert len(list(reader.next())) == 50
    assert len(list(reader.next())) == 50
    assert requested == [64]

    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 3)
    small = _make_reader(FormatType.PNG)
    _install(small, paths[:6])
    list(small.next())
    assert requested == [64, 3]


def test_epoch_change_uses_new_file_list(reader_args, monkeypatch):
    first = _write_files([3, 3, 3, 3], folder="/data/e1")
    second = _write_files([7, 7], folder="/data/e2")
    reader = _make_reader(FormatType.NPZ)
    submitted = _record_submits(reader, monkeypatch)
    _install(reader, first)
    assert len(list(reader.next())) == 2
    _install(reader, second)
    assert len(list(reader.next())) == 1
    assert submitted == first + second


def test_early_close_leaves_no_running_read(reader_args, monkeypatch):
    import threading

    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 3)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    paths = _write_files([9] * 8)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    running = {"n": 0}
    lock = threading.Lock()
    original = reader._dfs_read_one

    def _slow(path):
        with lock:
            running["n"] += 1
        try:
            time.sleep(0.05)
            return original(path)
        finally:
            with lock:
                running["n"] -= 1

    monkeypatch.setattr(reader, "_dfs_read_one", _slow)
    gen = reader.next()
    next(gen)
    gen.close()
    assert running["n"] == 0
    assert reader._total_objects_read == 1
    assert reader._total_bytes_read == 9
    assert reader._dfs_cache == {}


def test_read_error_raises_at_its_position(reader_args, monkeypatch):
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 4)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    paths = _write_files([5] * 6)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    original = reader._dfs_read_one

    def _failing(path):
        if path == paths[3]:
            raise OSError("injected read failure")
        return original(path)

    monkeypatch.setattr(reader, "_dfs_read_one", _failing)
    gen = reader.next()
    for _ in range(3):
        next(gen)
    with pytest.raises(OSError, match="injected read failure"):
        next(gen)
    assert reader._total_objects_read == 3


class _InterruptedWait(BaseException):
    pass


def test_interrupt_while_head_read_runs_leaves_no_running_read(reader_args, monkeypatch):
    import threading

    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 3)
    monkeypatch.setattr(reader_args, "batch_size", 1)
    paths = _write_files([9] * 6)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    running = {"n": 0}
    lock = threading.Lock()
    original = reader._dfs_read_one

    def _slow(path):
        with lock:
            running["n"] += 1
        try:
            time.sleep(0.1)
            return original(path)
        finally:
            with lock:
                running["n"] -= 1

    def _interrupt_head(path, future):
        if path == paths[0]:
            def _result(timeout=None):
                raise _InterruptedWait()
            future.result = _result

    monkeypatch.setattr(reader, "_dfs_read_one", _slow)
    _record_submits(reader, monkeypatch, on_submit=_interrupt_head)
    gen = reader.next()
    with pytest.raises(_InterruptedWait):
        next(gen)
    assert running["n"] == 0
    assert reader._total_objects_read == 0
    assert reader._dfs_cache == {}


def test_two_epochs_with_finalize_reuse_read_pool(reader_args, monkeypatch):
    paths = _write_files([6] * 4)
    reader = _make_reader(FormatType.NPZ)
    _install(reader, paths)
    assert len(list(reader.next())) == 2
    pool = reader._dfs_read_pool
    reader.finalize()
    assert reader._total_objects_read == 0
    _install(reader, paths)
    assert len(list(reader.next())) == 2
    assert reader._dfs_read_pool is pool
    assert reader._total_objects_read == 4


# -- Worker connection lifetime ------------------------------------------------

_KEEP_ALIVE = []


def _worker_child(mode, backing_dir, log_path):
    """Body of a spawned worker: open the worker client, then exit by ``mode``."""
    os.environ[dfs_client.TEST_BACKING_DIR_ENV] = backing_dir
    from dlio_benchmark.storage import dfs_testing

    def _log(line):
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    original_close = dfs_testing.DirBackedDfsClient.close

    def _close(self):
        was_connected = self.connected
        original_close(self)
        if was_connected:
            _log("close")

    dfs_testing.DirBackedDfsClient.close = _close

    original_close_all = dfs_client.close_all

    def _close_all():
        _log("close_all")
        original_close_all()

    dfs_client.close_all = _close_all

    client = dfs_client.get_worker_client(POOL, CONT)
    assert dfs_client.get_worker_client(POOL, CONT) is client

    if mode == "exception":
        raise RuntimeError("worker failed on purpose")
    if mode == "abandoned":
        def _reader():
            with client.open_read("/f.bin") as fh:
                while True:
                    yield fh.read(1)

        it = _reader()
        next(it)
        _KEEP_ALIVE.append(it)


def _run_worker(mode, tmp_path):
    backing = tmp_path / "backing"
    (backing / POOL / CONT).mkdir(parents=True)
    (backing / POOL / CONT / "f.bin").write_bytes(b"0123456789")
    log_path = tmp_path / "close.log"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_worker_child, args=(mode, str(backing), str(log_path)))
    proc.start()
    proc.join(timeout=120)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=10)
        pytest.fail(f"worker ({mode}) did not exit")
    lines = log_path.read_text().splitlines() if log_path.exists() else []
    return proc.exitcode, lines


def test_worker_close_once_on_normal_exit(tmp_path):
    exitcode, lines = _run_worker("normal", tmp_path)
    assert exitcode == 0
    assert lines == ["close_all", "close"]


def test_worker_close_on_exception(tmp_path):
    exitcode, lines = _run_worker("exception", tmp_path)
    assert exitcode == 1
    assert lines == ["close_all", "close"]


def test_worker_close_when_iterator_abandoned(tmp_path):
    exitcode, lines = _run_worker("abandoned", tmp_path)
    assert exitcode == 0
    assert lines == ["close_all", "close"]


def test_worker_init_opens_client_only_in_daos_workers(reader_args, monkeypatch):
    from dlio_benchmark.data_loader import torch_data_loader as tdl

    calls = []
    monkeypatch.setattr(dfs_client, "get_worker_client",
                        lambda *a: calls.append(a) or object())
    tdl._daos_worker_init(-1)
    assert calls == []
    tdl._daos_worker_init(3)
    assert calls == [(POOL, CONT, None)]
    monkeypatch.setattr(reader_args, "storage_type", StorageType.LOCAL_FS)
    tdl._daos_worker_init(4)
    assert calls == [(POOL, CONT, None)]


# -- TFRecord ------------------------------------------------------------------


def _make_tf_reader(reader_args, monkeypatch, thread_index=-1):
    from dlio_benchmark.reader.reader_factory import ReaderFactory
    monkeypatch.setattr(reader_args, "format", FormatType.TFRECORD)
    monkeypatch.setattr(reader_args, "data_loader", DataLoaderType.TENSORFLOW)
    return ReaderFactory.get_reader(type=FormatType.TFRECORD, dataset_type=DatasetType.TRAIN,
                                    thread_index=thread_index, epoch_number=1)


def test_factory_selects_dfs_reader_tfrecord(reader_args, monkeypatch):
    from dlio_benchmark.reader._dfs_iterable_mixin import _DfsIterableMixin
    from dlio_benchmark.reader.reader_handler import FormatReader
    reader = _make_tf_reader(reader_args, monkeypatch)
    assert type(reader).__name__ == "TFRecordReaderDfsIterable"
    assert type(reader).__bases__ == (FormatReader, _DfsIterableMixin)


def test_tfrecord_reader_prefetch_uses_dfs_client(reader_args, monkeypatch):
    monkeypatch.setattr(reader_args, "batch_size", 3)
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 2)
    paths = _write_files([11, 22, 33, 44, 55], ext="tfrecord")
    reader = _make_tf_reader(reader_args, monkeypatch)
    # Two thread slots, as TFDataLoader leaves them; one file has two samples.
    reader.file_map.clear()
    reader.file_map[0] = [(0, paths[0], 0), (1, paths[1], 0), (2, paths[1], 1)]
    reader.file_map[1] = [(3, paths[2], 0), (4, paths[3], 0), (5, paths[4], 0), (6, paths[0], 1)]

    opened = []
    original_open = reader._dfs_client.open_read

    def _spy_open(path):
        opened.append(path)
        return original_open(path)

    monkeypatch.setattr(reader._dfs_client, "open_read", _spy_open)
    sizes = []
    original_get_sample = reader.get_sample

    def _spy_sample(filename, sample_index):
        sizes.append(reader.open(filename))
        return original_get_sample(filename, sample_index)

    monkeypatch.setattr(reader, "get_sample", _spy_sample)

    batches = list(reader.next())
    assert sorted(opened) == sorted(paths)
    assert len(batches) == 7 // 3
    for batch in batches:
        assert batch.shape == (3,) + reader_args.resized_image.shape
    assert sorted(sizes) == sorted([11, 11, 22, 22, 33, 44, 55])
    assert reader._total_bytes_read == 11 + 22 + 33 + 44 + 55


def test_tfrecord_reader_never_opens_local_path(reader_args, monkeypatch):
    import builtins
    import io as _io

    paths = _write_files([8, 8, 8, 8], ext="tfrecord")
    reader = _make_tf_reader(reader_args, monkeypatch)
    reader.file_map.clear()
    reader.file_map[0] = [(i, p, 0) for i, p in enumerate(paths)]

    attempted = []
    real_open, real_io_open, real_os_open = builtins.open, _io.open, os.open

    def _guard(real):
        def _wrapped(file, *a, **k):
            attempted.append(os.fsdecode(file) if isinstance(file, (str, bytes, os.PathLike)) else file)
            return real(file, *a, **k)
        return _wrapped

    monkeypatch.setattr(builtins, "open", _guard(real_open))
    monkeypatch.setattr(_io, "open", _guard(real_io_open))
    monkeypatch.setattr(os, "open", _guard(real_os_open))
    batches = list(reader.next())
    assert len(batches) == 2
    assert not any(p in attempted for p in paths)


def test_tf_loader_builds_dfs_tfrecord_reader(reader_args, monkeypatch):
    pytest.importorskip("tensorflow")
    from dlio_benchmark.data_loader.tf_data_loader import TFDataLoader
    monkeypatch.setattr(reader_args, "format", FormatType.TFRECORD)
    monkeypatch.setattr(reader_args, "data_loader", DataLoaderType.TENSORFLOW)
    monkeypatch.setattr(reader_args, "read_threads", 2)
    loader = TFDataLoader(FormatType.TFRECORD, DatasetType.TRAIN, 1)
    loader.read()
    passes = loader._dataset
    assert type(passes).__name__ == "_DfsAllThreadsPasses"
    assert type(passes._reader).__name__ == "TFRecordReaderDfsIterable"


def _tf_all_threads_reader(reader_args, monkeypatch, sizes, folder="/data/train"):
    monkeypatch.setattr(reader_args, "batch_size", 2)
    monkeypatch.setitem(reader_args.storage_options, "prefetch_window", 2)
    paths = _write_files(sizes, folder=folder, ext="tfrecord")
    reader = _make_tf_reader(reader_args, monkeypatch)
    reader_args.train_file_map.clear()
    reader_args.train_file_map[0] = [(i, p, 0) for i, p in enumerate([q for n, q in enumerate(paths) if n % 2 == 0])]
    reader_args.train_file_map[1] = [(i, p, 0) for i, p in enumerate([q for n, q in enumerate(paths) if n % 2 == 1])]
    return reader, paths


def test_tfrecord_all_threads_result_is_reiterable(reader_args, monkeypatch):
    reader, _ = _tf_all_threads_reader(reader_args, monkeypatch, [8] * 6)
    passes = reader.next()
    assert len(list(passes)) == 3
    assert len(list(passes)) == 3
    assert iter(passes) is not iter(passes)


def test_tfrecord_abandoned_prewarm_pass_does_not_shorten_next(reader_args, monkeypatch):
    reader, _ = _tf_all_threads_reader(reader_args, monkeypatch, [8] * 6)
    passes = reader.next()
    warmup = iter(passes)
    next(warmup)
    warmup.close()
    assert reader._dfs_cache == {}
    assert len(list(passes)) == 3


def test_tfrecord_pass_reads_current_file_map(reader_args, monkeypatch):
    reader, first = _tf_all_threads_reader(reader_args, monkeypatch, [8] * 4)
    passes = reader.next()

    opened = []
    original_open = reader._dfs_client.open_read

    def _spy_open(path):
        opened.append(path)
        return original_open(path)

    monkeypatch.setattr(reader._dfs_client, "open_read", _spy_open)
    assert len(list(passes)) == 2
    assert sorted(opened) == sorted(first)

    second = _write_files([8] * 6, folder="/data/train2", ext="tfrecord")
    # reconfigure() installs a new dict rather than mutating the old one.
    monkeypatch.setattr(reader_args, "train_file_map",
                        {0: [(i, p, 0) for i, p in enumerate(second)]})
    opened.clear()
    assert len(list(passes)) == 3
    assert sorted(opened) == sorted(second)


# -- Parquet -------------------------------------------------------------------


def _write_parquet(path, rows=10, row_group_size=4):
    import io as _io
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "a": pa.array(range(rows), type=pa.int64()),
        "b": pa.array([os.urandom(32) for _ in range(rows)], type=pa.binary()),
    })
    sink = _io.BytesIO()
    pq.write_table(table, sink, row_group_size=row_group_size)
    client = dfs_client.get_client(POOL, CONT)
    client.mkdir(path.rsplit("/", 1)[0])
    client.write(path, sink.getvalue())
    return pq.ParquetFile(_io.BytesIO(sink.getvalue())).metadata


def _make_parquet_reader(reader_args, monkeypatch, thread_index=0):
    from dlio_benchmark.reader.reader_factory import ReaderFactory
    monkeypatch.setattr(reader_args, "format", FormatType.PARQUET)
    return ReaderFactory.get_reader(type=FormatType.PARQUET, dataset_type=DatasetType.TRAIN,
                                    thread_index=thread_index, epoch_number=1)


def test_factory_selects_dfs_reader_parquet(reader_args, monkeypatch):
    from dlio_benchmark.reader.reader_handler import FormatReader
    monkeypatch.setitem(reader_args.storage_options, "storage_library", "s3dlio")
    reader = _make_parquet_reader(reader_args, monkeypatch)
    assert type(reader).__name__ == "ParquetReaderDfsIterable"
    assert type(reader).__bases__ == (FormatReader,)
    assert reader._prefetch_executor._max_workers <= 64
    reader.finalize()


def test_torch_loader_selects_row_group_dataset_for_daos(reader_args, monkeypatch):
    from dlio_benchmark.data_loader.torch_data_loader import (
        TorchDataLoader, TorchIterableDataset,
    )
    monkeypatch.setattr(reader_args, "format", FormatType.PARQUET)
    monkeypatch.setattr(reader_args, "read_threads", 1)
    monkeypatch.setattr(reader_args, "multiprocessing_context", "spawn")
    loader = TorchDataLoader(FormatType.PARQUET, DatasetType.TRAIN, 1)
    loader.read()
    assert isinstance(loader._dataset.dataset, TorchIterableDataset)


def test_parquet_footer_opened_through_dfs_file(reader_args, monkeypatch):
    import pyarrow.parquet as pq

    path = "/data/pq/part_0.parquet"
    meta = _write_parquet(path, rows=10, row_group_size=4)
    reader = _make_parquet_reader(reader_args, monkeypatch)
    sources = []
    real_parquet_file = pq.ParquetFile

    def _spy(source, *a, **k):
        sources.append(source)
        return real_parquet_file(source, *a, **k)

    monkeypatch.setattr(pq, "ParquetFile", _spy)
    pf, cached_source, offsets = reader.open_footer_only(path)
    assert len(sources) == 1
    assert not isinstance(sources[0], (str, bytes, os.PathLike))
    assert hasattr(sources[0], "size")
    assert sources[0].closed
    assert cached_source is None
    assert pf.metadata.num_row_groups == meta.num_row_groups == 3
    assert offsets == [0, 4, 8, 10]
    assert reader.open_footer_only(path) is reader._pf_cache[path]
    assert len(sources) == 1
    reader.finalize()


def test_parquet_footer_handle_released_after_cache_pop(reader_args, monkeypatch):
    import io as _io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dlio_benchmark.storage.dfs_client import DfsClient
    from tests.test_dfs_client_binding import FakeLib

    fake = FakeLib()
    fake.add_dir("/pq")
    table = pa.table({"a": pa.array(range(10), type=pa.int64())})
    sink = _io.BytesIO()
    pq.write_table(table, sink, row_group_size=4)
    paths = ["/pq/a.parquet", "/pq/b.parquet"]
    for p in paths:
        fake.add_file(p, sink.getvalue())

    client = DfsClient(POOL, CONT, lib=fake)
    client.connect()
    reader = _make_parquet_reader(reader_args, monkeypatch)
    monkeypatch.setattr(reader, "_dfs_client", client)
    try:
        pf, cached_source, offsets = reader.open_footer_only(paths[0])
        assert cached_source is None
        assert len(client._handles) == 0
        # Metadata stays usable after the footer source is closed.
        assert offsets == [0, 4, 8, 10]
        assert reader._rg_byte_range(pf, 0)[1] > 0
        reader._pf_cache.pop(paths[0], None)

        reader.open(paths[1])
        for fut in list(reader._prefetch_futures.values()):
            assert fut.result() > 0
        reader._pf_cache.pop(paths[1], None)
        assert len(client._handles) == 0
        assert fake.count("dfs_release") > 0
    finally:
        reader.finalize()
        client.close()


def test_parquet_row_group_range_read(reader_args, monkeypatch):
    path = "/data/pq/part_1.parquet"
    meta = _write_parquet(path, rows=10, row_group_size=4)
    reader = _make_parquet_reader(reader_args, monkeypatch)

    calls = []
    original = reader._dfs_client.readinto

    def _spy(p, buf, offset=0):
        calls.append((p, len(buf), offset))
        return original(p, buf, offset)

    monkeypatch.setattr(reader._dfs_client, "readinto", _spy)

    expected = []
    for rg in range(meta.num_row_groups):
        rgm = meta.row_group(rg)
        starts, ends = [], []
        for c in range(rgm.num_columns):
            cm = rgm.column(c)
            s = cm.dictionary_page_offset if cm.dictionary_page_offset and cm.dictionary_page_offset > 0 \
                else cm.data_page_offset
            starts.append(s)
            ends.append(s + cm.total_compressed_size)
        comp = sum(rgm.column(c).total_compressed_size for c in range(rgm.num_columns))
        expected.append((min(starts), max(ends) - min(starts), comp))

    reader.open_file_map[path] = reader.open_footer_only(path)
    for rg in range(meta.num_row_groups):
        reader.submit_rg_prefetch(path, rg)
    _, _, offsets = reader.open_file_map[path]
    for rg in range(meta.num_row_groups):
        reader.get_sample(path, offsets[rg])

    assert sorted(calls, key=lambda c: c[2]) == [(path, n, s) for s, n, _ in expected]
    assert [reader._rg_cache[(path, rg)] for rg in range(meta.num_row_groups)] == \
        [comp for _, _, comp in expected]
    reader.finalize()


def test_parquet_rg_dataset_iterates_in_process(reader_args, monkeypatch):
    from dlio_benchmark.data_loader.torch_data_loader import TorchIterableDataset

    paths = ["/data/pq/a.parquet", "/data/pq/b.parquet"]
    for p in paths:
        _write_parquet(p, rows=10, row_group_size=4)
    monkeypatch.setattr(reader_args, "format", FormatType.PARQUET)
    monkeypatch.setattr(reader_args, "file_list_train", paths, raising=False)
    monkeypatch.setattr(reader_args, "batch_size", 3)
    dataset = TorchIterableDataset(FormatType.PARQUET, DatasetType.TRAIN, 1, 3, 0)
    assert type(dataset.reader).__name__ == "ParquetReaderDfsIterable"
    batches = list(iter(dataset))
    assert len(batches) == 20 // 3
    dataset.reader.finalize()

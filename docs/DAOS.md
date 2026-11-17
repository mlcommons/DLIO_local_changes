# DAOS storage backend (`storage_type: daos`)

DLIO can read and write datasets and checkpoints in a DAOS POSIX container
directly through `libdfs`, without a dfuse mount. Every I/O call goes from
the benchmark process to the DAOS client library; no kernel file system is
involved.

## Requirements

- A DAOS client installation on every node that runs DLIO: `libdfs.so`
  (and the `libdaos` libraries it links to) must be loadable by the dynamic
  linker, and the DAOS agent must be running.
- An existing pool and an existing container of type `POSIX`. DLIO never
  creates or destroys pools or containers; it only creates and removes
  files and directories inside the container. For example:

  ```bash
  daos container create <pool> <container> --type POSIX
  ```

- No extra Python package. `libdfs.so` is opened with `ctypes` only when
  `storage_type` is `daos`; other storage types never load it.

## Configuration

The connection target (pool, container) and the paths inside the container
are configured separately:

```yaml
storage:
  storage_type: daos
  storage_root: /              # base path inside the container
  storage_options:
    pool: <pool label or UUID>
    container: <container label or UUID>
    # sys_name: daos_server    # optional DAOS system name

dataset:
  data_folder: unet3d          # relative to storage_root
```

`pool` and `container` are required. If either is missing or empty, DLIO
stops before connecting to DAOS and names the missing key
(`storage.storage_options.pool` or `storage.storage_options.container`).

### Path resolution

All paths are absolute paths inside the container.

| Setting | Resolved as |
|---|---|
| `storage.storage_root` | container path; `/` is the container root |
| `dataset.data_folder` | relative paths are joined to `storage_root`; absolute paths are used as given |
| `checkpoint.checkpoint_folder` | resolved from the container root (independent of `storage_root`), both when writing checkpoints and when counting them before a read |

The same paths are used by data generation, file listing, training, and
checkpointing, so a dataset generated with a given configuration is found
again by a training run with the same configuration.

### Optional `storage_options`

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `sys_name` | unset | all | DAOS system name passed to `dfs_connect` |
| `prefetch_window` | 4 (npz/npy/jpeg/png/tfrecord), 64 (parquet) | readers | files read ahead of the consumer, one read thread each (at most 64 threads); for parquet, row groups in flight per DataLoader worker |
| `stagger_workers` | `true` | npz/npy/jpeg/png | delay the start of worker *k* by *k* x `computation_time`, as the other iterable readers do |
| `prefetch_workers` | 64 (max 64) | parquet | range-read threads per DataLoader worker |
| `columns` | all | parquet | read only the row-group bytes of these columns |

While one file is consumed, up to `prefetch_window` later files are being
read or wait to be consumed.  The byte counts reported for an epoch cover
the files whose samples were consumed; files read ahead and dropped by an
early stop are not counted.  A read error surfaces when the failing file
reaches the head of the queue, after the files before it were consumed.

Read-ahead threads waiting in libdfs poll for completion by default and can
keep cores busy.  Keep ranks per host x `reader.read_threads` x
`prefetch_window` near or below the number of cores of the client host.
`D_POLL_TIMEOUT` in the client environment sets the progress timeout of
these waits in microseconds.  It is converted to milliseconds by integer
division before it reaches the network layer, so values below 1000 still
poll without waiting; larger values allow the progress call to block where
the provider supports it.

## Supported formats

| `dataset.format` | Data loader | Reader | What is read |
|---|---|---|---|
| `npz`, `npy` | `pytorch` | `NPZReaderDfsIterable`, `NPYReaderDfsIterable` | every assigned file in full |
| `jpeg`, `png` | `pytorch` | `ImageReaderDfsIterable` | every assigned file in full |
| `tfrecord` | `tensorflow` | `TFRecordReaderDfsIterable` | every file assigned to the rank in full |
| `parquet` | `pytorch` | `ParquetReaderDfsIterable` | the compressed bytes of each row group (one range read per row group); the footer is parsed from a libdfs file object |

The readers measure storage I/O the same way as the other iterable readers
in DLIO: data is read into a buffer, only its byte count is kept, and
nothing is decoded. Files are read on a thread pool of at most 64 threads per
process (per DataLoader worker for PyTorch), one window of files ahead of
the training loop.

The PyTorch data loader uses `TorchIterableDatasetSimple` for npz, npy, jpeg
and png, and the row-group `TorchIterableDataset` for parquet, as it does
for object storage.

Data generation writes every supported format through the storage backend.
TFRecord generation writes each file to a local temporary file first and then
copies it into the container, as it does for object storage.

### Checkpointing

PyTorch checkpointing (`framework: pytorch`) is supported. The data and the
I/O pattern follow mlpstorage's `StreamingCheckpointing`, which DLIO uses for
`file` and `direct_fs`, with the same parameters:

- Data: one dgen-py generator per checkpoint file, with `dedup_ratio=1.0`
  and `compress_ratio=1.0` and its threads limited to the CPUs of the node
  divided by the ranks on the node. Every chunk is fresh, incompressible
  data. dgen-py is a dependency of DLIO; without it, saving a checkpoint
  fails with `ImportError`.
- Save: a producer thread fills 4 buffers of 32 MiB in turn while the
  calling thread writes the filled buffers at consecutive offsets through
  one write handle. The file is created once. A buffer is filled again only
  after its write has returned. No subprocess is used, because a forked
  process cannot reuse the DAOS client state of its parent.
- Load: the file is split into 4 contiguous blocks of equal size (the last
  one may be shorter). Each block is read by its own thread, through its own
  read handle, in 128 MiB pieces (4 x the write chunk size).

DFS writes are synchronous and DAOS has no client-side page cache, so no
separate flush is issued. The DAOS backend is selected even when
`mlpstorage_py` is installed.

### Rejected configurations

These are rejected before any DAOS call is made:

- `dataset.format` other than the formats above (for data generation,
  training, or evaluation)
- `reader.data_loader: native_dali`
- `reader.odirect: True`
- `dataset.format: tfrecord` for training or evaluation with a data loader
  other than `tensorflow`
- Parquet generation with `dataset.parquet.partition_by`
- Checkpointing with a framework other than `pytorch`

## Processes and connections

- Each process owns one DAOS connection per (pool, container, system name),
  shared by the storage objects, readers, and checkpointing of that process.
  Each MPI rank therefore has its own connection.
- The main process closes its connection when the benchmark finalizes.
  Any process that opened a connection and did not close it explicitly
  closes it when the process exits.
- PyTorch DataLoader workers open their own connection in the worker
  initialization function and close it when the worker process exits
  (normally, on an exception, or with an abandoned iterator). Reads still in
  flight are completed before the connection is closed.

### Spawned DataLoader workers

The main process connects to DAOS before the DataLoader workers start, and
DAOS client state does not survive `fork()`. With `storage_type: daos`,
`reader.multiprocessing_context: fork` (the default) is therefore switched to
`spawn` automatically, and a message is logged. Set
`reader.multiprocessing_context: spawn` (or `forkserver`) explicitly to
suppress the message.

## Testing without DAOS

Setting the environment variable `DLIO_DAOS_TEST_BACKING_DIR` to a local
directory replaces the DAOS client with a directory-backed stand-in:
container `<container>` of pool `<pool>` becomes the directory
`$DLIO_DAOS_TEST_BACKING_DIR/<pool>/<container>`, and `libdfs.so` is never
loaded. Because the stand-in is selected by an environment variable and
keeps its data on disk, spawned DataLoader workers and MPI ranks see the
same container contents as the main process.

```bash
export DLIO_DAOS_TEST_BACKING_DIR=/tmp/daos-backing
dlio_benchmark workload=unet3d_h100 \
  ++workload.storage.storage_type=daos \
  ++workload.storage.storage_root=/ \
  ++workload.storage.storage_options.pool=pool0 \
  ++workload.storage.storage_options.container=cont0 \
  ++workload.workflow.generate_data=True ++workload.workflow.train=False
```

The variable is meant for tests only; never set it for a measurement.

The DAOS tests run in the fast CI workflow:

| File | Covers |
|---|---|
| `tests/test_dfs_client_binding.py` | the `ctypes` binding against a fake `libdfs` (arguments, flags, errors, short reads, directory iteration across batches) |
| `tests/test_dfs_storage.py` | the storage backend and the pre-connect configuration check |
| `tests/test_dfs_readers.py` | reader selection, prefetch, and worker connection lifetime |
| `tests/test_dfs_checkpointing.py` | checkpoint backend selection, generated data, buffer reuse, error handling, and parallel block reads |
| `tests/dlio_daos_benchmark_test.py` | generate, list, train, and delete for every supported format, spawned workers, and checkpoint write and read through `DLIOBenchmark` |

## DAOS versions

The binding uses only these `libdfs` functions: `dfs_init`, `dfs_fini`,
`dfs_connect`, `dfs_disconnect`, `dfs_lookup`, `dfs_release`,
`dfs_get_size`, `dfs_read`, `dfs_write`, `dfs_open`, `dfs_mkdir`,
`dfs_iterate`, `dfs_remove`. It does not use `dfs_readdir` or `dfs_stat`,
whose arguments depend on the platform's `struct dirent` and `struct stat`
layouts. Their declarations, and the layouts of `d_iov_t`, `d_sg_list_t`
and `daos_anchor_t`, were compared between the DAOS tags below.

| DAOS | Status |
|---|---|
| 2.8 | declarations compared; run with every supported format and with checkpointing |
| 2.6 (v2.6.5.p3) | declarations compared; not run |

## Known limitations

- With PyTorch DataLoader workers (`reader.read_threads` > 0), the iterable
  dataset runs on persistent workers. Each worker keeps reading the file
  list it received when it started, also after the epoch changes and the
  files are reshuffled. The object-store iterable path behaves the same way
  (`persistent_workers=True`, and the file list is taken from the worker's
  reader).

## Troubleshooting

### DataLoader workers crash or hang when freeing read buffers

**Symptom.** With the verbs provider (`ofi+verbs;ofi_rxm`), a DataLoader
worker process crashes with a segmentation fault, or stops making progress,
when it frees a large read buffer. The stack shows libfabric's
`ofi_mr_cache_search` or `ofi_insert_rebalance`.

**Cause.** This happens when the libfabric memory registration (MR) cache is
enabled. DAOS disables the MR cache on verbs clients by default
(`FI_MR_CACHE_MAX_COUNT=0`), but it is enabled when an administrator sets
`CRT_MRC_ENABLE=1`, for example through `client_env_vars` in
`daos_server.yml`. It shows up in spawned worker processes, which do not
initialize MPI.

**Fix.** Set `CRT_MRC_ENABLE=0` in the client environment that runs the
benchmark. In multi-host runs, make sure the variable reaches every rank
(with Open MPI, `-x CRT_MRC_ENABLE`).

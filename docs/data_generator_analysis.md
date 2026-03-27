# DLIO Benchmark: Object Storage Integration — Analysis, Fixes, and Status

**Initial Analysis**: January 2026  
**Implementation Completed**: March 2026  
**Scope**: All 10 format generators, base class, storage factory, framework layer, readers, and config  
**Status**: ✅ All 8 testable formats passing full put+verify+get cycle against MinIO via s3dlio  

---

## Executive Summary

An initial analysis of `dlio_benchmark/data_generator/` identified multiple correctness bugs,
design deficiencies, and missing object storage support affecting all 10 format generators.
**All identified issues have been fixed.** Additionally, the full read path for object storage
was audited and repaired, covering the TensorFlow framework layer, storage factory routing,
S3 URI handling in the configuration layer, and three new parallel-prefetch readers.

### Problems found and fixed

| Area | Problem | Severity | Status |
|------|---------|----------|--------|
| All generators | `np.random.seed(10)` — all MPI ranks produce identical data | High | ✅ Fixed |
| `npz_generator.py` | `put_data(out_path_spec, output)` passes `BytesIO` object, not bytes | High | ✅ Fixed |
| 6 of 10 generators | No object storage support — local FS only | High | ✅ Fixed |
| `IndexedBinaryGenerator`, `ParquetGenerator` | Legacy global-state NumPy RNG bypasses dgen-py | Medium | ✅ Fixed |
| All generators | ~15 line loop boilerplate copy-pasted into every subclass | Medium | ✅ Refactored |
| `tf_framework.py` | All object storage I/O routed through `tf.io.gfile` (no S3 support) | High | ✅ Fixed |
| `storage_factory.py` | TensorFlow framework received `S3Storage` (double-mangled URIs) | High | ✅ Fixed |
| `config.py` `build_sample_map_iter()` | `os.path.abspath()` mangles S3 URIs to local paths | High | ✅ Fixed |
| `tfrecord_reader_s3_iterable.py` | `thread_index=-1` caused `KeyError` in single-reader mode | High | ✅ Fixed |
| `aistore_storage.py` | Import-time warning printed even when AIStore not being used | Low | ✅ Fixed |
| Missing S3 readers | CSV, HDF5, TFRecord had no S3-capable reader implementation | High | ✅ Added |
| Missing tests | No test suite validating generator quality or object store end-to-end | Medium | ✅ Added |

---

## 1. What Was Fixed: Data Generators

### 1.1 MPI Seed Bug (all generators)

**Problem**: Every generator called `np.random.seed(10)` unconditionally before its
generation loop. Because this seed is static, every MPI rank produced **identical files** —
completely defeating the purpose of distributed generation.

**Fix**: The seed was made rank-dependent. A `_file_seed()` method was added to the base
class (`DataGenerator.BASE_SEED + global_file_index`), giving each file a unique,
reproducible seed that varies across ranks. The legacy global `np.random.seed()` call was
removed from all 10 subclasses.

### 1.2 NPZ Object Storage Bug (`npz_generator.py`)

**Problem**: The `generate()` method passed the `io.BytesIO` buffer *object* to
`storage.put_data()` instead of its contents:
```python
# Broken:
self.storage.put_data(out_path_spec, output)       # passes BytesIO object
# Fixed:
self.storage.put_data(out_path_spec, output.getvalue())  # passes bytes
```
NPZ files written to object storage were silently corrupted on every run.

### 1.3 Missing Object Storage Support (6 of 10 generators)

**Problem**: HDF5, CSV, TFRecord, IndexedBinary, Synthetic, and Parquet generators wrote
only to local filesystem paths. Running with `storage_type: s3` either silently wrote to
local paths or raised errors.

**Fix**: All 6 generators were updated to use `io.BytesIO()` as the write target when
not on local FS, then call `storage.put_data(out_path_spec, output.getvalue())` after
each file. Key implementation details by format:

- **HDF5**: `h5py.File(io.BytesIO(), 'w', driver='core', backing_store=False)` writes an
  in-memory HDF5 file; `.getvalue()` yields valid HDF5 bytes.
- **CSV**: `df.to_csv(io.StringIO())` then `.encode('utf-8')` → bytes.
- **TFRecord**: `tf.io.TFRecordWriter` writes to a temp file via `tf.io.gfile` for local
  FS; for object storage, records are serialized to `io.BytesIO()` and uploaded.
- **IndexedBinary**: Moved from MPI collective I/O to standard `BytesIO` buffer for
  object storage paths.
- **Synthetic**: String content encoded to bytes via `io.BytesIO()`.
- **Parquet**: `pq.write_table(table, buf)` where `buf = pa.BufferOutputStream()`;
  `.getvalue().to_pybytes()` yields valid Parquet bytes for upload.

### 1.4 Legacy RNG and dgen-py Integration

**Problem**: `IndexedBinaryGenerator` and `ParquetGenerator` bypassed `gen_random_tensor()`
and called legacy `np.random.randint()` / `np.random.rand()` directly — roughly 55× slower
than dgen-py for large numeric arrays.

**Fix**: Both generators were updated to call `gen_random_tensor()` for all large numeric
data, flowing through dgen-py (Xoshiro256++ via Rust/PyO3) at 155× NumPy throughput.

### 1.5 Boilerplate Deduplication — `_generate_files()` Template Method

**Problem**: The same ~15-line loop (seed, RNG init, dimension extraction, progress,
BytesIO/path selection, `put_data`) was copy-pasted into every generator.

**Fix**: A `_generate_files(write_fn)` template method was added to `DataGenerator`.
Each subclass now passes a format-specific `write_fn` closure; the base class handles all
bookkeeping. The per-file seed is derived from a flowing numpy Generator (not arithmetic
`BASE_SEED + i`), eliminating adjacent-seed correlation artifacts.

---

## 2. What Was Fixed: Read Path

### 2.1 `tf_framework.py` — Object Storage I/O Rewrite

**Problem**: All `TFFramework` storage methods (`create_node`, `get_node`, `walk_node`,
`delete_node`, `put_data`, `get_data`, `isfile`) routed through `tf.io.gfile.*`. This does
not support `s3://` URIs without `tensorflow-io` installed, and was fragile even when
installed.

**Fix**: All 7 methods now detect object store URIs via `_is_object_store_uri()`:
```python
@staticmethod
def _is_object_store_uri(id):
    return id.startswith(("s3://", "gs://", "az://", "azureml://"))
```
When an object store URI is detected, operations dispatch directly to `s3dlio`:
- `put_data` → `s3dlio.put_bytes(id, data)`
- `get_data` → `bytes(s3dlio.get(id))`
- `walk_node` → `s3dlio.list(id)` (strips prefix to match `listdir()` contract)
- `delete_node` → `s3dlio.list(id)` + `s3dlio.delete()` per object
- `get_node` → `s3dlio.exists(id)` → `MetadataType.FILE`
- `create_node` → no-op for object stores (no real directories)
- `isfile` → `s3dlio.exists(id)`

Local paths continue to use `tf.io.gfile.*` unchanged.

### 2.2 `storage_factory.py` — TensorFlow Routing Fix

**Problem**: `StorageFactory.get_storage()` only returned `ObjStoreLibStorage` (direct
s3dlio) for `FrameworkType.PYTORCH`. TensorFlow workloads received `S3Storage`, which
routes through `framework.put_data()` — already a fully-qualified S3 URI — causing a
double-prefix bug that resulted in `service error` failures.

**Fix**:
```python
# Before:
if framework == FrameworkType.PYTORCH:
# After:
if framework in (FrameworkType.PYTORCH, FrameworkType.TENSORFLOW):
```

### 2.3 `config.py` — S3 URI Mangling in `build_sample_map_iter()`

**Problem**: `build_sample_map_iter()` called `os.path.abspath(file_list[file_index])`
unconditionally on every entry. `os.path.abspath("s3://bucket/path")` converts to
`/cwd/s3:/bucket/path` — a mangled local path. This caused `s3dlio.get_many()` to fail
with `service error` because the keys were invalid.

`get_global_map_index()` (the other map-building path) already had a `StorageType.LOCAL_FS`
guard. `build_sample_map_iter()` was missing the same guard.

**Fix**: Added the identical guard:
```python
if self.storage_type == StorageType.LOCAL_FS:
    abs_path = os.path.abspath(file_list[file_index])
else:
    abs_path = file_list[file_index]
```

### 2.4 `tfrecord_reader_s3_iterable.py` — `thread_index=-1` Handling

**Problem**: `TFDataLoader` creates the TFRecord reader with `thread_index=-1` (single-
reader mode). `reader_handler.py` does `self.file_map[self.thread_index]` — a direct key
lookup. The `file_map` is keyed `0..N-1` (thread partitions); `-1` is never a valid key,
causing `KeyError: -1`.

**Fix**: `TFRecordReaderS3Iterable.next()` explicitly handles `thread_index=-1` by
collecting all `file_map` values, consolidating unique object keys, prefetching via
`_prefetch()`, then yielding batches — bypassing the `file_map[-1]` lookup entirely.

### 2.5 `aistore_storage.py` — Silent Import (no warning)

**Problem**: An unconditional `logging.warning()` fired at module import time whenever the
AIStore SDK was not installed — even for workloads that never touched AIStore.

**Fix**: The warning was removed. `AISTORE_AVAILABLE = False` is set silently. A clear
`ImportError` with install instructions is raised inside `AIStoreStorage.__init__()` only
when a user actually tries to use AIStore.

---

## 3. New Readers Added

Three new S3-capable parallel-prefetch readers were added using the existing
`_S3IterableMixin` pattern:

| Reader | File | Extends |
|--------|------|---------|
| `CSVReaderS3Iterable` | `csv_reader_s3_iterable.py` | `CSVReader` + `_S3IterableMixin` |
| `HDF5ReaderS3Iterable` | `hdf5_reader_s3_iterable.py` | `HDF5Reader` + `_S3IterableMixin` |
| `TFRecordReaderS3Iterable` | `tfrecord_reader_s3_iterable.py` | `NPYReader` + `_S3IterableMixin` |

**Design principle** shared by all three (and the existing NPY/NPZ readers): these are
storage benchmarks — only the I/O transfer matters. Each reader fetches full objects via
`s3dlio.get_many()` and stores only the raw byte count (int) per object. No CSV parsing,
no h5py decoding, no TFRecord/protobuf deserialization — all pure CPU overhead irrelevant
to storage measurement.

`reader_factory.py` was updated to dispatch CSV, HDF5, and TFRECORD to their respective
S3 iterable readers when `storage_library=s3dlio` is configured.

---

## 4. New Tests Added

### `tests/test_data_generator_improvements.py` (24 tests)

Validates generator correctness properties:
- `gen_random_tensor` seed reproducibility and entropy
- `DataGenerator` class constants and static helpers (`_file_seed`, `_extract_dims`)
- RNG flow-through: same `rng` object produces different output on successive calls
- Format correctness: generate files, open with native library, verify dtype/shape/schema
- Data uniqueness: non-identical data within and across files
- Reader compatibility: generated files parsed by matching DLIO reader class

### `tests/test_s3dlio_object_store.py` (8 tests)

End-to-end object storage integration test suite (opt-in; requires live MinIO):
```bash
DLIO_S3_INTEGRATION=1 pytest tests/test_s3dlio_object_store.py -v
```
Exercises the full DLIOBenchmark workflow: generate → verify object count → train/read back.
Credentials loaded from `.env` with real environment variables taking priority.

---

## 5. All-Format Test Results

The shell-based end-to-end test (`tests/object-store/test_s3dlio_formats.py`) exercises
all formats in a full put+verify+get cycle against a live MinIO endpoint via s3dlio:

| Format | Generator | Reader | Status |
|--------|-----------|--------|--------|
| npy | `NpyGenerator` | `NPYReaderS3Iterable` | ✅ PASS |
| npz | `NpzGenerator` | `NPZReaderS3Iterable` | ✅ PASS |
| hdf5 | `HDF5Generator` | `HDF5ReaderS3Iterable` | ✅ PASS |
| parquet | `ParquetGenerator` | (parquet reader) | ✅ PASS |
| csv | `CsvGenerator` | `CSVReaderS3Iterable` | ✅ PASS |
| jpeg | `JpegGenerator` | (jpeg reader) | ✅ PASS |
| png | `PngGenerator` | (png reader) | ✅ PASS |
| tfrecord | `TfDataGenerator` | `TFRecordReaderS3Iterable` | ✅ PASS |

**8 / 8 formats passing.** All three test phases pass for each format:
1. **Generate** — objects written to MinIO bucket
2. **Verify** — expected object count confirmed via `s3dlio.list()`
3. **Train/Read** — objects fetched back via DLIOBenchmark training loop

---

## 6. File Change Summary

### Modified files

| File | Change Summary |
|------|----------------|
| `data_generator/data_generator.py` | Added `_generate_files()` template, `_file_seed()`, `_extract_dims()`; fixed rank-unique seeding |
| `data_generator/npy_generator.py` | Migrated to `_generate_files()` template |
| `data_generator/npz_generator.py` | Fixed `output.getvalue()` bug; migrated to `_generate_files()` |
| `data_generator/jpeg_generator.py` | Migrated to `_generate_files()` |
| `data_generator/png_generator.py` | Migrated to `_generate_files()` |
| `data_generator/hdf5_generator.py` | Added object storage support via `h5py` core driver; migrated to `_generate_files()` |
| `data_generator/csv_generator.py` | Added object storage support via `io.StringIO`; migrated to `_generate_files()` |
| `data_generator/tf_generator.py` | Added object storage support; migrated to `_generate_files()` |
| `data_generator/indexed_binary_generator.py` | Added object storage support; replaced legacy RNG with `gen_random_tensor()` |
| `data_generator/synthetic_generator.py` | Added object storage support |
| `data_generator/parquet_generator.py` | Added object storage support via `pyarrow.BufferOutputStream`; replaced legacy RNG with `gen_random_tensor()` |
| `framework/tf_framework.py` | Rewrote all 7 storage methods to dispatch to s3dlio for object store URIs |
| `storage/storage_factory.py` | Route `FrameworkType.TENSORFLOW` to `ObjStoreLibStorage` (same as PYTORCH) |
| `storage/aistore_storage.py` | Removed import-time warning; defer error to `__init__()` |
| `reader/reader_factory.py` | Route CSV, HDF5, TFRECORD to S3 iterable readers when `storage_library=s3dlio` |
| `utils/config.py` | Added `StorageType.LOCAL_FS` guard to `build_sample_map_iter()` to prevent `os.path.abspath()` mangling S3 URIs |
| `utils/utility.py` | Minor cleanup; dgen-py integration preserved |

### New files

| File | Purpose |
|------|---------|
| `reader/csv_reader_s3_iterable.py` | Parallel-prefetch CSV reader for S3 (s3dlio / s3torchconnector / minio) |
| `reader/hdf5_reader_s3_iterable.py` | Parallel-prefetch HDF5 reader for S3 |
| `reader/tfrecord_reader_s3_iterable.py` | Parallel-prefetch TFRecord reader for S3 (no protobuf decode) |
| `tests/test_data_generator_improvements.py` | 24 unit + integration tests for generator correctness |
| `tests/test_s3dlio_object_store.py` | 8 end-to-end object storage integration tests (opt-in) |
| `docs/data_generator_analysis.md` | This document |
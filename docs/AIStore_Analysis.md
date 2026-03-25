# AIStore Support: Gap Analysis and Rationalization Options

**Date:** March 2026  
**Scope:** `dlio_benchmark` — `StorageType.AISTORE` implementation vs. `StorageType.S3`  
**Status:** Analysis only — no code changes made

---

## Background

`dlio_benchmark` supports AIStore natively via its Python SDK (`aistore.sdk`),
as a distinct storage type (`storage_type: aistore`) separate from the generic
S3 object storage path (`storage_type: s3`).  The AIStore implementation was
added as a standalone code path, creating maintenance overhead and several gaps
compared to the fully-featured S3 path.  This document identifies those gaps and
presents rationalization options.

---

## Current Architecture: Three Separate Storage Handler Paths

```
StorageType.AISTORE  → AIStoreStorage      (aistore.sdk.Client — native SDK)
StorageType.S3       → ObjStoreLibStorage  (s3dlio / minio / s3torchconnector)
StorageType.S3 (old) → S3Storage           (legacy fallback, no storage_library)
```

**Files involved:**

| File | Role |
|------|------|
| `dlio_benchmark/storage/aistore_storage.py` | AIStore native SDK handler |
| `dlio_benchmark/storage/obj_store_lib.py` | S3 multi-library handler |
| `dlio_benchmark/storage/s3_storage.py` | Legacy S3 fallback |
| `dlio_benchmark/storage/storage_factory.py` | Routes `StorageType` → class |
| `dlio_benchmark/reader/reader_factory.py` | Routes format + storage type → reader |
| `dlio_benchmark/utils/config.py` | Validation and checkpoint auto-selection |
| `tests/dlio_aistore_benchmark_test.py` | AIStore unit tests (4 tests) |

### AIStoreStorage — What It Implements

`AIStoreStorage(DataStorage)` uses the official `aistore.sdk` and implements the
full `DataStorage` interface:

- `get_uri`, `create_namespace`, `get_node`, `walk_node`, `delete_node`
- `put_data(id, data)` — stores via `obj.get_writer().put_content(data)`;
  **no multipart, no offset/length** (a TODO comment exists in the code)
- `get_data(id, offset, length)` — supports byte-range reads via HTTP
  `Range` header (`obj.get_reader(byte_range=...)`)
- `_clean_key()` — strips both `s3://` and `ais://` URI schemes + bucket prefix
- Lazy bucket initialization via `@property bucket`

---

## Gap 1: Checkpointing — Silently Broken

### The Problem

In `config.py` `derive_configurations()`, checkpoint mechanism auto-selection
covers `StorageType.S3` correctly for **all three S3 libraries**, then falls
through to `PT_SAVE` for everything else — including AIStore:

```python
if self.checkpoint_mechanism == CheckpointMechanismType.NONE:
    elif self.framework == FrameworkType.PYTORCH:
        if self.storage_type == StorageType.S3:
            # s3torchconnector uses its native S3Checkpoint API (PT_S3_SAVE).
            # minio and s3dlio use the generic ObjStoreLib checkpoint (PT_OBJ_SAVE).
            storage_library = (self.storage_options or {}).get("storage_library", "s3torchconnector")
            if storage_library == "s3torchconnector":
                self.checkpoint_mechanism = CheckpointMechanismType.PT_S3_SAVE
            else:  # ← correctly covers both minio AND s3dlio
                self.checkpoint_mechanism = CheckpointMechanismType.PT_OBJ_SAVE
        else:
            # ← StorageType.AISTORE falls here — local filesystem checkpoint!
            self.checkpoint_mechanism = CheckpointMechanismType.PT_SAVE
```

**The S3 path is correct.** All three libraries are properly handled:
- `s3torchconnector` → `PT_S3_SAVE` (its native `S3Checkpoint` API)
- `minio` → `PT_OBJ_SAVE` (via the `else` branch; the code comment says so explicitly)
- `s3dlio` → `PT_OBJ_SAVE` (same `else` branch)

The validation block (around line 392) also explicitly names all three libraries
with separate `elif storage_library == "..."` branches and enforces the correct
mechanism for each.

**The gap is AIStore only.** When `storage_type: aistore` is configured with
`do_checkpoint: True`, the outer `if self.storage_type == StorageType.S3:` test
is `False`, so execution falls to the outer `else` and sets `PT_SAVE` —
**local filesystem checkpointing**.  No error is raised, no warning is logged.
The user believes they are testing AIStore checkpointing; they are testing
local-disk checkpointing.

### Validation Gap (same section)

The S3 validation block (same function) enforces for every library:
- SDK installed
- Credentials present (`access_key_id`, `secret_access_key`, `endpoint_url`)
- Format compatible with chosen library
- `checkpoint_mechanism` is the correct value

The AIStore block (lines 352–359) checks only:
- "Is the `aistore` SDK package installed?"

No credential/endpoint validation, no checkpoint-mechanism enforcement.

### Test Coverage Gap

The AIStore test file (`tests/dlio_aistore_benchmark_test.py`) contains
exactly **4 tests**:

| Test | What it covers |
|------|---------------|
| `test_aistore_gen_data` | Data generation (NPY, NPZ × PyTorch) |
| `test_aistore_train` | Training loop (NPY, NPZ × even/odd file counts) |
| `test_aistore_eval` | Evaluation pass |
| `test_aistore_multi_threads` | Multi-threaded reads (0, 1, 2 threads) |

**Checkpointing is not tested at all.**  The word "checkpoint" does not appear
anywhere in the AIStore test file.

---

## Gap 2: Reader Routing — Inconsistent Per Format

`reader_factory.py` includes AIStore in the same dispatch tuples as S3:

```python
elif _args.storage_type in (StorageType.S3, StorageType.AISTORE):
    storage_library = (getattr(_args, "storage_options", {}) or {}).get("storage_library")
    if storage_library in ("s3dlio", "s3torchconnector", "minio"):
        return NPYReaderS3Iterable(...)   # fast, streaming
    return NPYReaderS3(...)               # simpler fallback
```

For AIStore, `storage_library` is never set to `"s3dlio"`, `"s3torchconnector"`,
or `"minio"` (those are S3-path options).  The `if storage_library in (...)` check
always evaluates to `False` for AIStore.

The per-format result:

| Format | Reader selected for AIStore | Works? | Notes |
|--------|----------------------------|--------|-------|
| **NPY** | `NPYReaderS3` | ✅ | Uses `DataStorage.get_data()` → `AIStoreStorage.get_data()` |
| **NPZ** | `NPZReaderS3` | ✅ | Same abstract interface |
| **JPEG / PNG** | `ImageReader` (filesystem fallback) | ❌ | Falls out of the `storage_library in (...)` check; PIL-based filesystem reader cannot reach object storage |
| **PARQUET** | `ParquetReaderS3Iterable` | ⚠️ | Bypasses `DataStorage` entirely — calls S3 SDKs directly; defaults `storage_library` to `"s3dlio"`. May work if AIStore's S3-compatible endpoint is in use, but is completely untested and undocumented |
| **HDF5** | filesystem only | ❌ | Never supported object storage |
| **CSV** | filesystem only | ❌ | Never supported object storage |
| **TFRecord** | filesystem reader | ❌ | Never supported object storage |

**Performance discrepancy for NPY/NPZ:** S3 with a recognized library gets the
iterable, streaming readers (`NPYReaderS3Iterable`, `NPZReaderS3Iterable`).
AIStore always gets the simpler one-shot readers (`NPYReaderS3`, `NPZReaderS3`).
This is a performance gap, not a correctness bug.

---

## Gap 3: Validation Is One-Sided

The following validation is performed for `StorageType.S3` but **not** for
`StorageType.AISTORE`:

| Validation | S3 | AIStore |
|-----------|-----|---------|
| SDK installed | ✅ all three libraries | ✅ |
| `endpoint_url` required | ✅ raises if missing | ❌ not checked |
| `access_key_id` / `secret_access_key` | ✅ raises if missing | ❌ not checked |
| Format supported by library | ✅ (e.g. s3torchconnector → NPY/NPZ only) | ❌ not checked |
| `checkpoint_mechanism` is correct value | ✅ raises if wrong | ❌ not checked |

AIStore requires `endpoint_url` in `storage_options` (or `AWS_ENDPOINT_URL`) to
connect to the cluster, but the config validator does not enforce it.

---

## Full Feature Parity Summary

| Feature | S3 (ObjStoreLibStorage) | AIStore |
|---------|------------------------|---------|
| PyTorch checkpointing | ✅ PT_OBJ_SAVE or PT_S3_SAVE | ❌ silently falls back to PT_SAVE (local disk) |
| NPY / NPZ training | ✅ iterable + non-iterable | ⚠️ non-iterable only |
| JPEG / PNG training | ✅ ImageReaderS3Iterable | ❌ falls to filesystem ImageReader |
| Parquet training | ✅ explicit per-library byte-range | ⚠️ implicit s3dlio default, untested |
| HDF5 / CSV training | ❌ filesystem only | ❌ filesystem only |
| Config: checkpoint validation | ✅ enforces correct mechanism | ❌ none |
| Config: credential validation | ✅ checks access_key + endpoint | ❌ only SDK install |
| Test coverage: checkpointing | ✅ | ❌ zero |
| Test coverage: JPEG/PNG | ✅ | ❌ zero |

---

## Rationalization Options

### Option A — Route AIStore Through Its S3-Compatible Gateway *(simplest)*

AIStore exposes a standard S3-compatible HTTP endpoint.  Configure AIStore as
`storage_type: s3` with any of the three existing libraries (recommended:
`storage_library: s3dlio`):

```yaml
storage:
  storage_type: s3
  storage_root: my-ais-bucket
  storage_options:
    storage_library: s3dlio          # or minio
    endpoint_url: http://ais-host:8080
    access_key_id: ${AIS_ACCESS_KEY}
    secret_access_key: ${AIS_SECRET_KEY}
```

This eliminates `AIStoreStorage`, `storage_type: aistore`, and the entire
parallel code path.  Immediate full feature parity: checkpointing, all readers,
all libraries, all validation.

**Pros:**
- Zero new code
- Immediate full feature parity across all formats and checkpointing
- Reduced maintenance surface

**Cons:**
- Loses native SDK advantages (AIStore ETL jobs, server-side transforms,
  AIS-specific metadata APIs) — irrelevant for a benchmarking tool
- Existing `storage_type: aistore` YAML configs would need to change

---

### Option B — Fill the AIStore Gaps *(most consistent native-SDK path)*

Keep `storage_type: aistore` and the native SDK path; add the missing features:

1. **Checkpointing:** Add `StorageType.AISTORE` to the checkpoint auto-select
   block in `config.py` (e.g., `checkpoint_mechanism = PT_OBJ_SAVE`).  Verify
   that `PyTorchObjStoreCheckpointing` can work with `AIStoreStorage` as the
   backend, or implement a thin `PT_AIS_SAVE` mechanism.

2. **JPEG/PNG reader:** Add `storage_type: aistore` awareness — either create
   `ImageReaderAIS` that calls `storage.get_data()`, or route AIStore through
   the existing iterable reader with an `AIStoreStorage` adapter.

3. **Parquet reader:** Add an `_AISRangeFile` equivalent that wraps
   `AIStoreStorage.get_data(offset=, length=)` (the method already supports
   byte-range reads) so Parquet row-group reads go through the abstract
   interface.

4. **Validation:** Add endpoint and credential checks for AIStore in `config.py`,
   matching what exists for S3.

5. **Tests:** Add checkpoint tests to `dlio_aistore_benchmark_test.py`; add
   JPEG/PNG training tests.

**Pros:**
- Keeps native SDK path with AIStore-unique capabilities
- No changes to existing AIStore YAML configs

**Cons:**
- Every new reader or feature must be implemented twice (once for S3, once for AIStore)
- Tests must be maintained in two places
- This is why the current gaps exist — Option B requires ongoing discipline

---

### Option C — Consolidate AIStore Into ObjStoreLibStorage as a 4th Library *(cleanest long-term)*

Add `storage_library: aistore` as a fourth option inside `ObjStoreLibStorage`.
The user's YAML uses `storage_type: s3` (unchanged for all existing S3 user
configs) and sets `storage_library: aistore`.  Internally, `ObjStoreLibStorage`
dispatches to the `aistore.sdk` when `storage_library == "aistore"`.

```yaml
storage:
  storage_type: s3
  storage_root: my-ais-bucket
  storage_options:
    storage_library: aistore
    endpoint_url: http://ais-host:8080
```

The reader factory naturally handles AIStore through `(StorageType.S3, ...)` with
a `storage_library == "aistore"` branch where needed.  Checkpoint auto-select
in `config.py` is already `StorageType.S3`-gated, so it would automatically apply.

**Pros:**
- One storage class handles all object storage backends
- All reader routing, checkpoint, and validation logic is unified immediately
- Adding `storage_library in (..., "aistore")` to the iterable-reader checks
  gives NPY/NPZ the faster streaming reader for free
- Tests have one parametrized fixture, not two

**Cons:**
- Moderate refactor — `AIStoreStorage` class is deleted, logic folded into
  `ObjStoreLibStorage`
- The `storage_type: aistore` config key is deprecated; users must update YAMLs
  (mitigatable with a deprecation shim in `storage_factory.py`)

---

## Recommendation

For a **benchmarking tool** (as opposed to an application that uses AIStore
ETL or server-side transformations), either **Option A** or **Option C**
eliminates the maintenance burden with no functional loss:

- **Option A** is zero-code but requires users to change their YAML configs.
  Best if there are few AIStore users and fast resolution is the goal.
- **Option C** is the architecturally cleanest long-term solution, preserves
  the ability to add AIStore-SDK-specific optimizations later (e.g. AIStore
  prefetch hints), and allows configs to keep `storage_type: aistore` with a
  shim.
- **Option B** is only justified if there is a concrete need for AIStore native
  SDK features (ETL, server-side transforms) that cannot be exposed through the
  S3-compatible gateway.

**Option B (patch-and-continue) should be avoided** unless the native SDK
features are actively needed — every gap that exists today is a direct result
of the current two-path strategy.

---

## Files to Change by Option

### Option A (S3 Gateway — no new code)

| Change | File |
|--------|------|
| Update user-facing docs to recommend `storage_type: s3, storage_library: s3dlio` for AIStore | `docs/AIStore_Analysis.md`, `docs/STORAGE_LIBRARIES.md` |
| Mark `storage_type: aistore` deprecated | `dlio_benchmark/storage/storage_factory.py`, `dlio_benchmark/common/enumerations.py` |

### Option B (Fill Gaps)

| Change | File |
|--------|------|
| Add AIStore to checkpoint auto-select | `dlio_benchmark/utils/config.py` |
| Add AIStore validation (endpoint, credentials, mechanism) | `dlio_benchmark/utils/config.py` |
| Fix JPEG/PNG reader routing for AIStore | `dlio_benchmark/reader/reader_factory.py` |
| Add `_AISRangeFile` wrapper for Parquet | `dlio_benchmark/reader/parquet_reader_s3_iterable.py` |
| Add checkpoint and JPEG/PNG tests | `tests/dlio_aistore_benchmark_test.py` |

### Option C (Consolidate)

| Change | File |
|--------|------|
| Fold `AIStoreStorage` into `ObjStoreLibStorage` as `storage_library: aistore` | `dlio_benchmark/storage/obj_store_lib.py` |
| Add `storage_type: aistore` shim → `storage_type: s3, storage_library: aistore` | `dlio_benchmark/storage/storage_factory.py` |
| Update `config.py` validation to include `storage_library: aistore` branches | `dlio_benchmark/utils/config.py` |
| Add `"aistore"` to iterable-reader `storage_library` checks | `dlio_benchmark/reader/reader_factory.py` |
| Delete `dlio_benchmark/storage/aistore_storage.py` | — |
| Migrate AIStore tests to use S3 parametrized fixture | `tests/dlio_aistore_benchmark_test.py` |

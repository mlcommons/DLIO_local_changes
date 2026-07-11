# S3DLIO_RT_THREADS sentinel-propagation fix (DLIO v3.0.3, 2026-07-11)

**Status: RESOLVED.** The DLIO-side sentinel-propagation bug (this doc)
and the complementary s3dlio-side sanity clamp (v0.9.112) together
recover full multipart-upload throughput. Either fix on its own is
sufficient; both together are the clean end state.

## Symptom

`mlpstorage whatif training unet3d datagen object` — driving DLIO's NPZ
generator through `s3dlio` into a local s3-ultra target — ceilinged at
~200 MB/s per rank, while isolated s3dlio benchmarks against the same
s3-ultra hit 2500 MB/s per rank. Same code path, same wire, ~10×
throughput gap.

## Root cause

`ObjStoreLibStorage.__init__` (in [`dlio_benchmark/storage/obj_store_lib.py`](../dlio_benchmark/storage/obj_store_lib.py))
derived `S3DLIO_RT_THREADS` from `self._args.write_threads` at
storage-construction time:

```python
_write_threads = getattr(self._args, "write_threads", 8)
_rt_threads = min(_write_threads * 3 // 2, 128)
os.environ["S3DLIO_RT_THREADS"] = str(_rt_threads)
```

The intent (from the surrounding comment block) was to hint s3dlio's
Tokio async runtime at a worker count matched to this rank's actual
concurrency, so that N MPI ranks on the same host wouldn't each
independently claim the full core count.

The timing subtlety: `ObjStoreLibStorage.__init__` runs at
[`dlio_benchmark/main.py:132`](../dlio_benchmark/main.py#L132), BEFORE
[`dlio_benchmark/main.py:505`](../dlio_benchmark/main.py#L505)
`self.args.derive_configurations()` auto-sizes `write_threads` from
its dataclass default sentinel (`1`) to the real workload-appropriate
value (e.g. 32 for a 28-core NP=1 S3 workload).

So at the moment `S3DLIO_RT_THREADS` was being set:

```
_write_threads  = 1        # the auto-size sentinel — not the real value
_rt_threads     = min(1 * 3 // 2, 128) = 1
os.environ["S3DLIO_RT_THREADS"] = "1"
```

s3dlio faithfully obeyed the env var and built its global Tokio runtime
with **one** worker. Every concurrent multipart-upload part serialized
on that worker: p50 `close()` = 22 843 ms/file, aggregate ~214 MB/s
per rank.

## Fix (DLIO side, this repo)

Extract the S3DLIO_RT_THREADS block into `_configure_s3dlio_runtime_env()`
and skip the auto-derive when `write_threads` is still at the auto-size
sentinel:

```python
_write_threads = getattr(self._args, "write_threads", 1)
if _write_threads <= 1:
    # Also clear a stale ancestor auto-set (e.g. parent mlpstorage
    # process auto-set to 1 based on its own not-yet-finalized
    # write_threads).  Leaving that in place would preserve the poison;
    # unsetting it hands sizing back to s3dlio's own MPI-aware default.
    os.environ.pop("S3DLIO_RT_THREADS", None)
    os.environ.pop("_S3DLIO_RT_AUTO", None)
    return
```

s3dlio v0.9.112+ handles MPI-aware auto-sizing natively via
`_pymod`'s `configure_thread_pools(0)` at import time (one worker per
available core divided by the MPI world size), which is exactly the
target the downstream `write_threads * 1.5` derivation would have
landed near anyway.

For explicit user configs (`write_threads > 1` in YAML) and legitimate
ancestor-recompute (`write_threads > 1` with `_S3DLIO_RT_AUTO=1` set),
the `1.5 × write_threads` (cap 128) derivation still applies unchanged.

## Complementary fix (s3dlio side, v0.9.112)

s3dlio v0.9.112 additionally clamps env-var values below
`RT_THREADS_LIMIT / 4` up to `RT_THREADS_LIMIT` (with a stderr warning
the first time it fires). This is defense-in-depth for any downstream
library that miscomputes `S3DLIO_RT_THREADS` — the DLIO fix eliminates
the specific miscomputation source in this repo, but s3dlio's clamp
catches equivalent bugs in any other caller.

## RED-then-GREEN verification

RED evidence (2026-07-11 investigation session): a temporary diagnostic
logfile inside `s3dlio::s3_client::global_rt_handle()` captured, at the
first S3 op:

```
[pid=381939] configure_thread_pools(input=0) resolved=28 ...
[pid=381939] global_rt_handle: get_runtime_threads()=1
             S3DLIO_RT_THREADS=Some("1") ...
```

That's the pre-fix `ObjStoreLibStorage.__init__` writing `"1"` to the
env var and s3dlio picking it up as-is.

GREEN evidence: five regression tests in
[`tests/test_s3dlio_rt_threads_sentinel.py`](../tests/test_s3dlio_rt_threads_sentinel.py)
cover every relevant path — sentinel path leaves env unset, explicit
`write_threads > 1` still auto-derives, user pre-set is respected
regardless, ancestor-recompute path still overwrites when
`write_threads > 1`, and the sentinel-on-both corner case actively
clears a stale ancestor value. All 5 pass; full checkin suite
(`tests/test_fast_ci.py`) still passes 92/92 unchanged.

## Live before/after (real s3-ultra, NP=1, 40 × 146 MB files)

| Configuration                                    | Wall time | Throughput  |
| ------------------------------------------------ | --------- | ----------- |
| Pre-fix s3dlio + pre-fix DLIO                    | 27.4 s    | 214 MB/s    |
| Fixed s3dlio (v0.9.112 clamp) + pre-fix DLIO     | 2.9 s     | ~1928 MB/s  |
| Fixed s3dlio (v0.9.112) + fixed DLIO (v3.0.3)    | 3.0 s     | ~1955 MB/s  |

In the "clamp only" row, s3dlio's warning `s3dlio: S3DLIO_RT_THREADS=1
is < RT_THREADS_LIMIT/4 (limit=28); using 28 instead …` fires (clamp
catches DLIO's bad env value). In the "both fixed" row, the warning
does NOT fire because DLIO no longer sets the bad env value in the
first place.

## Cross-repo references

- s3dlio investigation doc: `../../s3dlio/docs/investigation/DLIO_UNET3D_DATAGEN_BOTTLENECK_INVESTIGATION_2026-07-10.md`
- s3dlio Changelog v0.9.112: `../../s3dlio/docs/Changelog.md`
- s3dlio `_configure_thread_pools` auto-init call site: `../../s3dlio/src/lib.rs`
- s3dlio `clamped_env_rt_threads` helper: `../../s3dlio/src/s3_client.rs`

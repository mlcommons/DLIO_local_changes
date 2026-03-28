# DLIO Benchmark I/O Issues — Executive Summary

**Date:** 2026-03-28  
**Full technical document:** [DLIO_IO_Issues-Proposal_2026-03-28.md](DLIO_IO_Issues-Proposal_2026-03-28.md)  
**Audience:** Engineering leads, project owners, and decision-makers who need to understand the scope of issues and the investment required to address them — without implementation details.

---

## What This Review Found

A code review of the `dlio_benchmark` codebase identified thirteen distinct issues across data generation, data loading, checkpointing, configuration management, and benchmark correctness. The most significant finding is that **results produced by the current codebase for local-filesystem and object-storage workloads are not directly comparable to each other**, because the two backend paths perform different amounts of CPU work even when given identical data. This calls into question a class of published comparisons.

The issues range from critical correctness bugs to structural inefficiencies. All are actionable. None require redesigning the benchmark's overall architecture.

---

## Critical Issues (Affect Correctness of Results)

### 1. File and Object Storage Backends Are Not Measuring the Same Thing

The object-storage readers were written to skip all data decoding — they read raw bytes, record the byte count, and discard the bytes, because DLIO returns a pre-allocated random tensor to the training loop regardless of what was read. The local-filesystem readers were not updated to match: they fully decode every JPEG file (using PIL), fully load every NPY array (using NumPy), and fully inflate compressed HDF5 datasets — all of which is then discarded.

**Consequence:** A local-filesystem JPEG benchmark spends 70–99% of training-step time on CPU image decoding, not on I/O. An equivalent object-storage benchmark spends near 0% on decoding. The same storage hardware running the same data through the two paths can produce benchmark numbers that differ by 5–20× due entirely to this CPU overhead difference, not actual storage performance differences.

**Decision required:** Bring local-filesystem readers up to the standard already implemented in the S3 iterable readers. This is a code-only change and does not affect the storage I/O being measured. Until this is done, cross-backend comparisons in benchmark reports are not internally consistent.

→ Full analysis: [Section 13](DLIO_IO_Issues-Proposal_2026-03-28.md#13-file-vs-object-workload-asymmetry--closing-the-performance-gap)

### 2. Data Generation Is Slower Than It Needs to Be by Orders of Magnitude

JPEG and PNG data generation is CPU-bottlenecked on image compression, not on storage write throughput. At typical image sizes, generating an ImageNet-scale dataset (1.28 million files) takes approximately 80 minutes per rank for JPEG, and over 4 hours per rank for PNG. The actual storage write takes roughly 16 seconds per rank. Generation time is 300–1000× longer than storage write time, dominated entirely by compression work that has no bearing on the storage being benchmarked.

For the most common benchmark configurations (non-DALI data loaders), JPEG and PNG files do not need to be valid image files, because the reader never decodes them. The generator can write raw random bytes directly, collapsing generation overhead from ~30 milliseconds per file to under 0.01 milliseconds — a 2000–4000× speedup. This applies to all configurations except those using NVIDIA DALI, which calls a real image decoder and therefore requires valid JPEG bitstreams.

**Decision required:** Update JPEG and PNG generators to detect the configured data loader and skip image encoding when the reader does not decode. For DALI configurations, accept the encoding cost as unavoidable and document it as a known constraint.

→ Full analysis: [Section 9g](DLIO_IO_Issues-Proposal_2026-03-28.md#9g-jpeGpng-do-files-need-to-be-actually-valid-images), [Section 9d](DLIO_IO_Issues-Proposal_2026-03-28.md#9d-where-time-actually-goes-in-an-end-to-end-jpeg-benchmark-run)

### 3. TFRecord / Iterative Sampler Reads the Wrong Files on Non-Zero Ranks

A file-index tracking bug in `build_sample_map_iter()` causes MPI rank 1 and above to read from the wrong portion of the dataset when using the iterative data sampler (standard for TFRecord workloads). The first file read per rank is correct; all subsequent reads revert to iterating from the beginning of the file list. Both rank 0 and rank 1 end up reading the same overlapping set of files while the upper half of the dataset is never read by any rank.

**Consequence:** Any TFRecord benchmark result using more than one MPI rank double-counts data from the lower half of the dataset and misses the upper half entirely. Reported throughput is inflated and not reproducible by other means.

**Decision required:** Fix the file-index counter in `build_sample_map_iter()`. The PyTorch index sampler does not have this bug.

→ Full analysis: [Section 2b](DLIO_IO_Issues-Proposal_2026-03-28.md#2b-tf--iterative-path--build_sample_map_iter-used-when-data_loader_sampler--iterative), [Section 6e](DLIO_IO_Issues-Proposal_2026-03-28.md#6e-build_sample_map_iter-bug--concrete-description)

---

## High-Priority Issues (Significantly Affect Benchmark Quality)

### 4. `read_threads` Is Hardcoded at a Value That Is Wrong at Scale

The thread count for parallel I/O is set as a fixed integer in each YAML config file and is never adjusted for the actual deployment topology. For JPEG/PNG workloads, storage throughput scales directly with the number of concurrent open requests. With the default value, a typical NFS deployment uses less than 10% of its available bandwidth — not because the storage is slow, but because the benchmark is not issuing enough concurrent requests. The correct value varies by an order of magnitude depending on how many MPI ranks share a node.

**Decision required:** Support an `auto` setting for `read_threads` that resolves at runtime based on the actual MPI topology. Keep the integer form for reproducible runs. Update default configs to a higher starting value.

→ Full analysis: [Section 11](DLIO_IO_Issues-Proposal_2026-03-28.md#11-read_threads--fixed-yaml-value-vs-runtime-adaptive-sizing)

### 5. Deduplicating Storage Systems Will Produce Meaningless Results Without Unique File Content

Every generated file must contain content that is byte-unique across the entire dataset. Storage systems from major enterprise vendors (NetApp, Pure Storage, Vast Data, and many object stores) apply inline deduplication by default. If multiple files share identical byte content, the storage system physically stores only one copy and the benchmark measures deduplication throughput rather than storage write throughput. Results can appear orders of magnitude higher than the system's actual sustainable ingestion rate.

The codebase correctly uses a unique random seed per file via dgen-py; however, any shortcut that pre-computes one serialized blob and copies it across files — for any format — would silently produce deduplicated data. This constraint must be treated as non-negotiable for any benchmark run on production storage.

→ Full analysis: [Section 9e](DLIO_IO_Issues-Proposal_2026-03-28.md#9e-the-non-negotiable-constraint-every-file-must-contain-unique-bytes)

### 6. Storage Reader CPU Overhead Contaminates Training-Step Timing

Even apart from the file/object asymmetry described in Issue 1, all local-filesystem readers include CPU decode time inside the training-step latency window. The benchmark reports this combined time as if it were pure storage access time. For JPEG workloads, 71–99% of the reported per-sample time is CPU decoding, not storage I/O.

→ Full analysis: [Section 9c](DLIO_IO_Issues-Proposal_2026-03-28.md#9c-reader-overhead-by-format-local-filesystem-path), [Section 9d](DLIO_IO_Issues-Proposal_2026-03-28.md#9d-where-time-actually-goes-in-an-end-to-end-jpeg-benchmark-run)

---

## Structural Issues (Reduce Maintainability and Reproducibility)

### 7. Forty-Nine Configuration Files for a Small Orthogonal Matrix

The `configs/dlio/workload/` directory contains 49 YAML files covering a matrix of approximately 7 models × 4 storage backends × 2–3 phases. The file count grows multiplicatively with every new backend or model. Files share 90–95% identical content; the differing fields are storage backend name, bucket name, and endpoint URL. The endpoint URLs hard-code a specific lab IP address, making every object-storage config file non-portable outside that lab.

Hydra, the configuration framework already in use, supports config composition through config groups. Adopting it reduces the 49 files to approximately 13 (7 model configs plus 3 shared storage templates plus 3 workflow configs), with connection details supplied at runtime rather than baked into files.

→ Full analysis: [Section 7](DLIO_IO_Issues-Proposal_2026-03-28.md#7-yaml-config-proliferation-analysis), [Section 8](DLIO_IO_Issues-Proposal_2026-03-28.md#8-proposed-yaml-config-architecture)

### 8. `multiprocessing_context` Must Match the Storage Library or Hangs Silently

The fork-vs-spawn setting for DataLoader workers must be `spawn` for object-storage libraries that maintain background threads (s3dlio, s3torchconnector). If a user copies a local-filesystem YAML and adds an object-storage backend without changing `multiprocessing_context`, all object-storage reads will silently hang with no error message. The constraint is documented only in YAML comments, not enforced in code.

→ Full analysis: [Section 6c](DLIO_IO_Issues-Proposal_2026-03-28.md#6c-multiprocessing_context-couples-to-storage_library-but-lives-in-reader)

### 9. `storage_library` Config Schema Is Inconsistent

The `storage_library` field lives in an inconsistent location across the YAML schema, dataclass, and validation code. This creates ambiguity in how CLI overrides are expressed and silently returns `None` in any code path that accesses the field outside the standard load sequence.

→ Full analysis: [Section 6a](DLIO_IO_Issues-Proposal_2026-03-28.md#6a-storage_library-promotion-inconsistency)

---

## Lower-Priority Issues (Operational Efficiency)

### 10. No Intra-Rank Parallelism for Data Generation

Each MPI rank generates files sequentially. On multi-core nodes, all cores beyond the one doing the generation loop sit idle during what is usually the longest phase of a benchmark run. Adding thread-level parallelism within each rank would multiply generation throughput by the available core count.

→ Full analysis: [Section 5, Item 2](DLIO_IO_Issues-Proposal_2026-03-28.md#5-specific-improvement-opportunities), [Section 12e, Item 3](DLIO_IO_Issues-Proposal_2026-03-28.md#12e-recommendations)

### 11. Object Store Generation Has No Async Pipeline

Each file is generated and uploaded synchronously. Generation and upload cannot overlap, meaning each rank waits for the upload acknowledgment before generating the next file. An async upload pipeline would allow the CPU to generate the next file while the network transfers the previous one.

→ Full analysis: [Section 5, Item 4](DLIO_IO_Issues-Proposal_2026-03-28.md#5-specific-improvement-opportunities)

### 12. MPI Topology Is Collected but Not Used for Resource Planning

DLIO already collects per-node rank counts and node indices at startup, but does not use this information to auto-size thread counts, assign file-locality by node, or report topology in benchmark output. All three uses are straightforward given the existing data.

→ Full analysis: [Section 12](DLIO_IO_Issues-Proposal_2026-03-28.md#12-mpi-multi-host-topology--available-infrastructure-missing-integration)

### 13. No Settle-Time Guard After Generation on Eventual-Consistency Systems

After data generation completes, the benchmark immediately begins listing the generated files. On object stores with eventual-consistency semantics or NFS with attribute caching, newly written objects may not be visible to a listing immediately. If the listing returns fewer files than expected, the benchmark aborts with an error rather than retrying.

→ Full analysis: [Section 6f](DLIO_IO_Issues-Proposal_2026-03-28.md#6f-no-barrier-before-directory-walk-in-initialize)

---

## Recommended Prioritization

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| **Immediate** | File vs. object reader asymmetry (Issue 1) | Medium | Invalidates cross-backend comparisons |
| **Immediate** | TFRecord iterative sampler bug (Issue 3) | Low | Invalidates multi-rank TFRecord results |
| **High** | JPEG/PNG generator skips encoding for non-DALI (Issue 2) | Medium | Reduces generation from hours to seconds |
| **High** | Unique-bytes constraint enforcement (Issue 5) | Low | Prevents meaningless results on dedup storage |
| **High** | Auto-size `read_threads` (Issue 4) | Low | Unlocks full storage bandwidth at scale |
| **Medium** | Derive `multiprocessing_context` automatically (Issue 8) | Low | Prevents silent hangs on config copy/paste |
| **Medium** | YAML config composition with Hydra (Issue 7) | High | Reduces maintenance burden by ~70% |
| **Medium** | Intra-rank generation parallelism (Issue 10) | Medium | Reduces generation wall-clock time proportionally |
| **Low** | Async object-store upload pipeline (Issue 11) | Medium | Marginal throughput improvement |
| **Low** | Node-local file affinity and topology logging (Issue 12) | Low | Improves NFS locality and result reproducibility |
| **Low** | Post-generation settle time (Issue 13) | Low | Prevents spurious failures on object stores |

---

## What Is Already Working Well

The following design decisions in the current codebase are correct and should be preserved:

- **dgen-py for data generation**: the zero-copy Rust-backed PRNG is the right foundation for all format generators. It is fast enough to never be the bottleneck and produces genuinely unique content per file.
- **S3 iterable readers**: the skip-decode architecture is correct and complete. The task is to apply the same pattern to local-filesystem readers, not to change the object-storage path.
- **Per-rank checkpoint files**: the distributed checkpointing design (each rank writes its own file, no serialization, barriers only at epoch boundaries) is correct for the workload being simulated.
- **MPI topology collection in DLIOMPI**: the infrastructure to make topology-aware decisions is already present. It only needs to be wired into resource planning.
- **TFRecord reader**: already returns the pre-allocated tensor without touching file bytes — the correct behaviour that all other readers need to adopt.

---

*Full technical analysis, code examples, and implementation details are in [DLIO_IO_Issues-Proposal_2026-03-28.md](DLIO_IO_Issues-Proposal_2026-03-28.md).*

"""
NPZ reader using parallel prefetch from S3-compatible object storage.
See _s3_iterable_mixin.py for the full design rationale.

Three storage libraries are supported (strictly isolated, no cross-library fallback):
  s3dlio           — s3dlio.get_many(), up to 64 parallel requests, O(1) len(BytesView)
  s3torchconnector — S3IterableDataset.from_objects() + sequential reader
  minio            — ThreadPoolExecutor + Minio SDK, pooled TCP connections

Only the raw byte count is stored per object — no numpy decode.
"""
# Copyright (c) 2025, UChicago Argonne, LLC. Apache 2.0 License.
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.npz_reader import NPZReader
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class NPZReaderS3Iterable(NPZReader, _S3IterableMixin):
    """
    Parallel-prefetch NPZ reader for S3-compatible object stores.

    All prefetch, library routing, and byte-counting logic is in _S3IterableMixin.
    This class is a thin adapter connecting the mixin to DLIO's FormatReader chain.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._s3_init(opts)
        self.logger.info(
            f"{utcnow()} NPZReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    @dlp.log
    def open(self, filename):
        return self._object_cache.get(filename)

    @dlp.log
    def close(self, filename):
        self._object_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        # Report byte count for telemetry. Do NOT call super() — NPZReader.get_sample()
        # does open_file_map[filename][..., sample_index].nbytes which would fail
        # because open_file_map[filename] is now an int (byte count), not an array.
        dlp.update(image_size=self._object_cache.get(filename, 0))

    def next(self):
        self._s3_prefetch_all()
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._s3_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

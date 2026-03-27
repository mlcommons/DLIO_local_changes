"""
HDF5 reader using parallel prefetch from S3-compatible object storage.
See _s3_iterable_mixin.py for the full design rationale.

This is a storage benchmark — we measure how fast bytes can be fetched from
object storage. h5py decoding is pure CPU overhead that adds noise to the
measurement and is never needed: FormatReader.next() always yields
self._args.resized_image, not the actual file contents.

This reader stores only the raw byte count (int) per object, exactly like
NPYReaderS3Iterable and NPZReaderS3Iterable. No h5py, no data decoding.

Three storage libraries are supported (strictly isolated, no cross-library fallback):
  s3dlio           — s3dlio.get_many(), up to 64 parallel requests
  s3torchconnector — S3IterableDataset.from_objects() + sequential reader
  minio            — ThreadPoolExecutor + Minio SDK, pooled TCP connections
"""
# Copyright (c) 2025, UChicago Argonne, LLC. Apache 2.0 License.
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.hdf5_reader import HDF5Reader
from dlio_benchmark.reader._s3_iterable_mixin import _S3IterableMixin
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class HDF5ReaderS3Iterable(HDF5Reader, _S3IterableMixin):
    """
    Parallel-prefetch HDF5 reader for S3-compatible object stores.

    Fetches every assigned object in parallel and stores only the raw byte
    count (int) — no h5py decoding. get_sample() reports that byte count as
    the image_size telemetry metric. The actual I/O transfer is fully measured;
    the omitted decode step is pure CPU overhead irrelevant to storage benchmarking.

    open_file_map[filename] holds an int (byte count), same pattern as
    NPYReaderS3Iterable / NPZReaderS3Iterable.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)
        opts = getattr(self._args, "storage_options", {}) or {}
        self._s3_init(opts)
        self.logger.info(
            f"{utcnow()} HDF5ReaderS3Iterable [{self._storage_library}] "
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
        # Report byte count for telemetry. Do NOT call super() — HDF5Reader.get_sample()
        # tries to index open_file_map[filename] as an h5py.File, which would fail
        # because open_file_map[filename] is now an int (byte count).
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

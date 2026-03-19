"""
   Copyright (c) 2025, UChicago Argonne, LLC
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
NPY reader using parallel/streaming fetch from object storage.

Mirrors npz_reader_s3_iterable.py for the NPY format.  The only difference
is that NPY files contain a single array (no named key), so decode is simply
np.load(BytesIO(data)) rather than np.load(BytesIO(data))['x'].

See npz_reader_s3_iterable.py for full design rationale and documentation.
"""
import io
import os
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.npy_reader import NPYReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class NPYReaderS3Iterable(NPYReader):
    """
    Parallel-prefetch NPY reader for S3-compatible object stores.

    Replaces the sequential get_data()-per-file pattern of NPYReaderS3 with a
    parallel prefetch of all files assigned to this DLIO worker thread.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index, epoch)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._opts = opts
        self._epoch = epoch
        self._file_cache = {}  # filename → np.ndarray, populated in next()

        if self._storage_library in ("s3dlio", "s3torchconnector"):
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        self.logger.info(
            f"{utcnow()} NPYReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    def _uri_for_filename(self, filename: str) -> str:
        if "://" in filename:
            return filename
        root = self._args.storage_root.rstrip("/")
        return f"s3://{root}/{filename.lstrip('/')}"

    def _prefetch_s3dlio(self, filenames: list) -> dict:
        import s3dlio

        uris = [self._uri_for_filename(f) for f in filenames]
        uri_to_fname = dict(zip(uris, filenames))
        results = s3dlio.get_many(uris)

        cache = {}
        for uri, data in results:
            fname = uri_to_fname.get(uri, uri)
            cache[fname] = np.load(io.BytesIO(bytes(data)), allow_pickle=True)
        return cache

    def _prefetch_minio(self, filenames: list) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        from urllib.parse import urlparse
        from minio import Minio

        opts = self._opts
        endpoint = opts.get("endpoint_url", "")
        if endpoint.startswith("https://"):
            host = endpoint[8:]
            secure = True
        elif endpoint.startswith("http://"):
            host = endpoint[7:]
            secure = False
        else:
            host = endpoint
            secure = False

        client = Minio(
            host,
            access_key=opts.get("access_key_id"),
            secret_key=opts.get("secret_access_key"),
            secure=secure,
            region=opts.get("region", "us-east-1"),
        )

        def _fetch_one(filename):
            uri = self._uri_for_filename(filename)
            parsed = urlparse(uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            resp = client.get_object(bucket, key)
            try:
                raw = resp.read()
            finally:
                resp.close()
                resp.release_conn()
            return filename, np.load(io.BytesIO(raw), allow_pickle=True)

        n_workers = min(16, max(1, len(filenames)))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fname, arr in pool.map(_fetch_one, filenames):
                cache[fname] = arr
        return cache

    def _prefetch(self, filenames: list) -> dict:
        lib = self._storage_library
        if lib in ("s3dlio", "s3torchconnector"):
            return self._prefetch_s3dlio(filenames)
        elif lib == "minio":
            return self._prefetch_minio(filenames)
        else:
            raise ValueError(
                f"NPYReaderS3Iterable: unknown storage_library {lib!r}; "
                f"supported: s3dlio, s3torchconnector, minio"
            )

    @dlp.log
    def open(self, filename):
        return self._file_cache.get(filename)

    @dlp.log
    def close(self, filename):
        self._file_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)

    def next(self):
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        filenames = []
        for _, fname, _ in thread_entries:
            if fname not in seen:
                seen.add(fname)
                filenames.append(fname)

        if filenames:
            self.logger.info(
                f"{utcnow()} NPYReaderS3Iterable thread={self.thread_index} "
                f"prefetching {len(filenames)} files via [{self._storage_library}]"
            )
            self._file_cache = self._prefetch(filenames)

        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        if filename not in self._file_cache:
            self._file_cache.update(self._prefetch([filename]))
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

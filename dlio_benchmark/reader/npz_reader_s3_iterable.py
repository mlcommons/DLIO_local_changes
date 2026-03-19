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
NPZ reader using parallel/streaming fetch from object storage, as opposed to
the sequential per-file pattern in NPZReaderS3.

Supported libraries:
  s3dlio          — uses s3dlio.get_many() (parallel, up to 64 in-flight requests)
  s3torchconnector — same as s3dlio (uses s3dlio as the underlying engine)
  minio           — uses concurrent.futures.ThreadPoolExecutor

All files assigned to this DLIO thread are fetched in parallel before iteration
begins, eliminating the serial latency of one S3 round-trip per file.

The reader integrates cleanly with DLIO's existing file_map / FormatReader
pipeline: open(filename) simply returns the pre-fetched array from the cache,
and get_sample / next / read_index all work through the standard parent chain.
"""
import io
import os
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.npz_reader import NPZReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class NPZReaderS3Iterable(NPZReader):
    """
    Parallel-prefetch NPZ reader for S3-compatible object stores.

    Replaces the sequential get_data()-per-file pattern of NPZReaderS3 with a
    parallel prefetch of all files assigned to this DLIO worker thread, using
    whichever storage library is configured via storage_options.storage_library.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        # NPZReader.__init__ → FormatReader.__init__ sets up file_map, thread_index, etc.
        # It does NOT create a storage connection, so it is safe to call here.
        super().__init__(dataset_type, thread_index, epoch)

        args = self._args
        opts = getattr(args, "storage_options", {}) or {}
        self._storage_library = opts.get("storage_library", "s3dlio")
        self._opts = opts
        self._epoch = epoch
        self._file_cache = {}  # filename → np.ndarray, populated in next()

        # Configure endpoint for s3dlio / s3torchconnector at construction time
        # so that any lazy import inside get_many picks it up immediately.
        if self._storage_library in ("s3dlio", "s3torchconnector"):
            ep = opts.get("endpoint_url")
            if ep and not os.environ.get("AWS_ENDPOINT_URL_S3"):
                os.environ["AWS_ENDPOINT_URL_S3"] = ep

        self.logger.info(
            f"{utcnow()} NPZReaderS3Iterable [{self._storage_library}] "
            f"thread={thread_index} epoch={epoch}"
        )

    # ── URI helpers ──────────────────────────────────────────────────────────

    def _uri_for_filename(self, filename: str) -> str:
        """Return a full s3:// URI for a DLIO filename (relative or absolute)."""
        if "://" in filename:
            return filename
        root = self._args.storage_root.rstrip("/")
        return f"s3://{root}/{filename.lstrip('/')}"

    # ── Parallel prefetch per library ────────────────────────────────────────

    def _prefetch_s3dlio(self, filenames: list) -> dict:
        """Fetch all filenames in parallel using s3dlio.get_many()."""
        import s3dlio

        uris = [self._uri_for_filename(f) for f in filenames]
        uri_to_fname = dict(zip(uris, filenames))

        # get_many() returns a list of (uri, BytesView) tuples, all fetched
        # concurrently with up to max_in_flight=64 outstanding requests.
        results = s3dlio.get_many(uris)

        cache = {}
        for uri, data in results:
            fname = uri_to_fname.get(uri, uri)
            cache[fname] = np.load(io.BytesIO(bytes(data)), allow_pickle=True)["x"]
        return cache

    def _prefetch_minio(self, filenames: list) -> dict:
        """Fetch all filenames concurrently using Minio SDK + ThreadPoolExecutor."""
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
            return filename, np.load(io.BytesIO(raw), allow_pickle=True)["x"]

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
                f"NPZReaderS3Iterable: unknown storage_library {lib!r}; "
                f"supported: s3dlio, s3torchconnector, minio"
            )

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """Return the pre-fetched array from the cache (no I/O at this point)."""
        return self._file_cache.get(filename)

    @dlp.log
    def close(self, filename):
        # Evict from cache to free memory once DLIO is done with this file.
        self._file_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        # Delegates to NPZReader.get_sample which reads self.open_file_map[filename]
        # (already populated by FormatReader.next via open()) and updates dlp metrics.
        super().get_sample(filename, sample_index)

    def next(self):
        """Pre-fetch all this thread's files in parallel, then yield batches."""
        thread_entries = self.file_map.get(self.thread_index, [])
        # Preserve order but deduplicate filenames (each file may contain multiple samples)
        seen = set()
        filenames = []
        for _, fname, _ in thread_entries:
            if fname not in seen:
                seen.add(fname)
                filenames.append(fname)

        if filenames:
            self.logger.info(
                f"{utcnow()} NPZReaderS3Iterable thread={self.thread_index} "
                f"prefetching {len(filenames)} files via [{self._storage_library}]"
            )
            self._file_cache = self._prefetch(filenames)

        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        """For ON_DEMAND reads: fetch a single file on demand if not cached."""
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

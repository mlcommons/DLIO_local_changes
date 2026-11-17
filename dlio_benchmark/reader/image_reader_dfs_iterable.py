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
JPEG/PNG reader for storage_type: daos (libdfs, no dfuse).

Each image file holds exactly one sample.

Each file is read in full through libdfs and only its byte count is kept;
nothing is decoded.  See _dfs_iterable_mixin.py for the prefetch design.

Inherits FormatReader + _DfsIterableMixin only, never the local-filesystem
reader, so no code path can open() a container path as a local file.
"""
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader._dfs_iterable_mixin import _DfsIterableMixin
from dlio_benchmark.utils.utility import Profile, dft_ai, utcnow

dlp = Profile(MODULE_DATA_READER)


class ImageReaderDfsIterable(FormatReader, _DfsIterableMixin):
    """Parallel-prefetch reader for DAOS (storage_type: daos)."""

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._dfs_init()
        self.logger.info(
            f"{utcnow()} ImageReaderDfsIterable thread={thread_index} epoch={epoch}"
        )

    @dlp.log
    def open(self, filename):
        return self._dfs_cache.get(filename)

    @dlp.log
    def close(self, filename):
        self._dfs_cache.pop(filename, None)

    @dlp.log
    def get_sample(self, filename, sample_index):
        byte_count = self._dfs_cache.get(filename, 0)
        dlp.update(image_size=byte_count)
        dft_ai.update(image_size=byte_count)

    def next(self):
        yield from self._dfs_stream_next()

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._dfs_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self.finalize_dfs_bytes()
        self._dfs_shutdown()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

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
TFRecord reader for storage_type: daos (libdfs, no dfuse).

TFDataLoader builds a single reader with ``thread_index=-1`` for TFRecord,
keeps the object returned by ``next()`` for the whole run and iterates it
once per epoch (and once more for the worker pre-warm).  In that mode
``next()`` returns a re-iterable object: every ``iter()`` starts a new pass
over the files assigned to this rank (all thread slots of the current
``train_file_map`` / ``val_file_map``), read through libdfs with in-order
read-ahead.  Only byte counts are kept, and one batch of ``args.resized_image``
is yielded per ``batch_size`` samples.  TFRecord protobufs are never parsed
and tensorflow is not imported.

Inherits FormatReader + _DfsIterableMixin only, never TFReader, whose
``tf.data.TFRecordDataset`` would open container paths as local files.
"""
from collections import Counter

import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader._dfs_iterable_mixin import _DfsIterableMixin
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class _DfsAllThreadsPasses:
    """Re-iterable result of ``next()`` for ``thread_index=-1``.

    Each ``iter()`` returns a new generator, so the same object can be
    iterated once per epoch.  Closing a generator early (an abandoned
    iterator) waits for its running reads.
    """

    def __init__(self, reader):
        self._reader = reader

    def __iter__(self):
        return self._reader._dfs_all_threads_next()


class TFRecordReaderDfsIterable(FormatReader, _DfsIterableMixin):
    """Parallel-prefetch TFRecord reader for DAOS (storage_type: daos)."""

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._dfs_init()
        self.logger.info(
            f"{utcnow()} TFRecordReaderDfsIterable thread={thread_index} epoch={epoch}"
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

    def next(self):
        if self.thread_index == -1:
            return _DfsAllThreadsPasses(self)
        return self._dfs_stream_next()

    def _current_file_map(self):
        """The file map as of now; ``reconfigure()`` replaces it every epoch."""
        if self.dataset_type is DatasetType.TRAIN:
            return self._args.train_file_map
        return self._args.val_file_map

    def _dfs_all_threads_next(self):
        """One pass over every thread slot; one ``(batch_size, ...)`` array per batch."""
        entries = []
        for thread_entries in (self._current_file_map() or {}).values():
            entries.extend(thread_entries or [])
        if not entries:
            return

        samples_per_path = Counter(filename for _, filename, _ in entries)
        paths = list(samples_per_path)
        window = self._dfs_window_size()
        self.logger.info(
            f"{utcnow()} TFRecordReaderDfsIterable thread=-1 "
            f"streaming {len(paths)} files ({len(entries)} samples), read-ahead {window} [daos]"
        )

        batch = None
        sample_buf = 0
        files = self._dfs_readahead(paths, window)
        try:
            for path, nbytes in files:
                self._dfs_cache = {path: nbytes}
                for s in range(samples_per_path[path]):
                    self.get_sample(path, s)
                    sample_buf += 1
                    if sample_buf == self.batch_size:
                        if batch is None:
                            batch = np.array([self._args.resized_image] * self.batch_size)
                        yield batch
                        sample_buf = 0
                self._dfs_cache = {}
        finally:
            files.close()
            self._dfs_cache = {}
        # Drop-last: a remaining partial batch is discarded.

    @dlp.log
    def read_index(self, image_idx, step):
        filename, _ = self.global_index_map[image_idx]
        self._dfs_ensure_cached(filename)
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self.finalize_dfs_bytes()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

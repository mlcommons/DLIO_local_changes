"""Lazy file index containers for DLIO datasets.

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

For large-scale runs the file list can grow into millions of entries.
A plain ``list[str]`` of paths costs ~210 bytes per entry (Python ``str``
header + UTF-8 bytes).  When PyTorch DataLoader workers ``spawn`` from
each rank they each get a full copy, exploding to hundreds of GB per
node and OOM-killing the run.

This module provides :class:`SyntheticFileList` as a drop-in replacement
for the legacy ``list[str]`` returned by ``storage.walk_node()``.

It is O(1) per lookup and *stateless* (the only resident memory is a
handful of scalar slots).  Works whenever the dataset layout matches what
``data_generator.py`` would produce, which covers every MLPerf Storage
closed/open workload that calls ``datagen``.  Memory cost does NOT scale
with ``num_files`` -- randomisation of access order is delegated to the
per-rank ``VirtualIndexMap._sample_list`` (which is sized to
``samples_per_rank``, not ``num_files_total``).

It exposes the minimum ``Sequence`` protocol the rest of DLIO uses
(``__len__``, ``__getitem__``, ``__iter__``, slicing) plus an in-place
``shuffle(seed)`` method (no-op for SyntheticFileList since shuffling is
handled at the sample-index level).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared with data_generator.py for path layout consistency
# ---------------------------------------------------------------------------

def _zero_pad_width(value: int) -> int:
    """Replicates ``len(str(n))`` used by data_generator.add_padding()."""
    return len(str(value))


# ---------------------------------------------------------------------------
# SyntheticFileList
# ---------------------------------------------------------------------------

class SyntheticFileList:
    """Lazy file path list for DLIO-generated datasets.

    Paths are formatted on demand from the same template used by
    ``dlio_benchmark.data_generator.data_generator``::

        {data_folder}/{dataset_type}/{i % num_subfolders}/{file_prefix}_{i}_of_{N}.{file_format}

    When ``num_subfolders <= 1`` the subfolder segment is omitted.  The
    object is stateless: no per-instance arrays are allocated -- the only
    memory cost is the scalar slots.

    The object is a faithful read-only ``Sequence[str]``; existing
    callers that index it, iterate it, or take ``len()`` work unchanged.
    Slicing returns a lightweight view that shares the parent's scalar
    fields and owns a sliced permutation.
    """

    __slots__ = (
        "_data_folder",
        "_dataset_type",
        "_length",
        "_num_files_total",
        "_num_subfolders",
        "_file_prefix",
        "_file_format",
        "_file_index_width",
        "_subfolder_index_width",
        "_view_start",
        "_view_stride",
        "_root_dir",
    )

    def __init__(self,
                 data_folder: str,
                 dataset_type: str,
                 num_files: int,
                 num_subfolders: int,
                 file_prefix: str = "img",
                 file_format: str = "npz") -> None:
        # Coerce enum-like inputs (DatasetType, FormatType) to plain
        # strings -- DLIO's config passes these as enum objects whose
        # repr matches their value (e.g. FormatType.JPEG -> 'jpeg').
        self._data_folder = str(data_folder)
        self._dataset_type = str(dataset_type)
        # `_num_files_total` is the dataset size that lives forever in
        # the `_of_N` suffix of every path (datagen burns this number
        # into filenames).  `_length` is the logical length of *this
        # view*; slicing shrinks it without changing the suffix.
        self._num_files_total = int(num_files)
        self._length = int(num_files)
        self._num_subfolders = int(num_subfolders)
        self._file_prefix = str(file_prefix)
        self._file_format = str(file_format)
        self._file_index_width = (
            _zero_pad_width(self._num_files_total)
            if self._num_files_total > 0 else 1
        )
        self._subfolder_index_width = (
            _zero_pad_width(self._num_subfolders)
            if self._num_subfolders > 0 else 1
        )
        self._root_dir = f"{self._data_folder}/{self._dataset_type}"
        # Stateless layout: file at view position ``p`` is
        # ``_format_path(_view_start + p * _view_stride)``.  No
        # per-instance arrays -- the only memory cost is the slots.
        # Slicing creates a sibling view that adjusts the start/stride
        # without copying anything.
        #
        # Shuffling is intentionally NOT handled here.  Randomisation
        # of access order is delegated to the per-rank
        # ``VirtualIndexMap._sample_list``, which already holds an
        # ``int64`` permutation of size ``samples_per_rank`` (not
        # ``num_files_total``).  Keeping shuffle state out of this
        # object means RAM does NOT grow with the total dataset size --
        # only with the per-rank slice -- which is what makes large
        # multi-node MLPerf submissions feasible.
        self._view_start = 0
        self._view_stride = 1

    # -- Sequence protocol ------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for position in range(self._length):
            yield self._format_path(
                self._view_start + position * self._view_stride
            )

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            view = SyntheticFileList.__new__(SyntheticFileList)
            view._data_folder = self._data_folder
            view._dataset_type = self._dataset_type
            view._num_files_total = self._num_files_total
            view._num_subfolders = self._num_subfolders
            view._file_prefix = self._file_prefix
            view._file_format = self._file_format
            view._file_index_width = self._file_index_width
            view._subfolder_index_width = self._subfolder_index_width
            view._root_dir = self._root_dir
            view._view_start = self._view_start + start * self._view_stride
            view._view_stride = self._view_stride * step
            view._length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
            return view
        position = int(index)
        if position < 0:
            position += self._length
        if not 0 <= position < self._length:
            raise IndexError(
                f"SyntheticFileList index {index} out of range [0, {self._length})"
            )
        return self._format_path(
            self._view_start + position * self._view_stride
        )

    def __repr__(self) -> str:
        return (f"SyntheticFileList(num_files={self._num_files_total}, "
                f"len={self._length}, "
                f"view_start={self._view_start}, "
                f"view_stride={self._view_stride}, "
                f"num_subfolders={self._num_subfolders}, "
                f"root={self._root_dir!r}, "
                f"prefix={self._file_prefix!r}, "
                f"format={self._file_format!r})")

    # -- Mutators ---------------------------------------------------------

    def shuffle(self, seed: int) -> None:
        """No-op: shuffling is delegated to ``VirtualIndexMap._sample_list``.

        This object is intentionally stateless so its memory footprint
        does NOT scale with ``num_files_total``.  A file_shuffle setting
        in the workload YAML is satisfied indirectly: the rank's
        ``_sample_list`` (size = samples_per_rank, not num_files_total)
        is shuffled by ``sample_shuffle``, and because for
        ``samples_per_file=1`` we have ``file_index == sample_index``,
        the resulting access order is already uniformly random.

        If both ``file_shuffle`` and ``sample_shuffle`` are OFF the
        access order is the identity, which matches legacy behaviour
        when only ``file_shuffle`` is set (since the stateless view
        cannot honour it locally).  Callers that need a randomised
        order with ``sample_shuffle=OFF`` must enable ``sample_shuffle``.
        """
        # Accept the seed argument for API compatibility with the
        # legacy ``list[str]`` path and ``shuffle_file_list`` helper,
        # but deliberately do nothing.
        _ = int(seed)

    # -- Internals --------------------------------------------------------

    def _format_path(self, file_index: int) -> str:
        """Render the path for *logical* file index ``file_index``.

        ``file_index`` is the value stored in ``_permutation`` -- the
        file's identity on disk.  Matches ``data_generator.py``'s
        ``file_spec`` exactly; in particular the ``_of_N`` suffix
        always reflects the original dataset size
        (``_num_files_total``), not the length of any slice view.
        """
        if self._num_subfolders > 1:
            subfolder = (
                f"{file_index % self._num_subfolders:0{self._subfolder_index_width}d}"
            )
            return (
                f"{self._root_dir}/{subfolder}/"
                f"{self._file_prefix}_"
                f"{file_index:0{self._file_index_width}d}"
                f"_of_{self._num_files_total}.{self._file_format}"
            )
        return (
            f"{self._root_dir}/"
            f"{self._file_prefix}_"
            f"{file_index:0{self._file_index_width}d}"
            f"_of_{self._num_files_total}.{self._file_format}"
        )

    # -- Disk discovery ---------------------------------------------------

    @classmethod
    def peek_total_from_disk(cls,
                             data_folder: str,
                             dataset_type: str,
                             num_subfolders: int,
                             file_prefix: str = "img",
                             file_format: str = "npz",
                             expected_total: int = 0,
                             storage=None) -> int:
        """Discover or verify the ``_of_N`` suffix baked into filenames.

        When ``expected_total`` is provided (the common case — it comes
        from the workload YAML's ``num_files_train``), this verifies
        that file index 0 with that N exists using a single
        ``storage.get_node()`` call and returns immediately.

        When ``expected_total`` is 0, falls back to reading a single
        directory entry from the first subfolder to parse N out of the
        filename.  This fallback uses ``os.scandir`` (local FS only).

        Args:
            data_folder: root of the dataset.
            dataset_type: ``"train"`` or ``"valid"``.
            num_subfolders: same value used at datagen time.
            file_prefix: same value used at datagen time.
            file_format: same value used at datagen time.
            expected_total: if non-zero, verify file 0 exists with this
                N and return immediately.  If zero, auto-discover N
                from a single directory entry (local FS only).
            storage: a storage handler exposing ``get_node(path)`` that
                returns non-None if the path exists.  When None, falls
                back to ``os.path.exists``.

        Returns:
            The total number of files baked into the filename suffix.

        Raises:
            FileNotFoundError: if no matching files are present or
                ``expected_total`` does not match on-disk layout.
        """
        data_folder = str(data_folder)
        dataset_type = str(dataset_type)
        file_prefix = str(file_prefix)
        file_format = str(file_format)

        def _exists(path: str) -> bool:
            if storage is not None:
                return storage.get_node(path) is not None
            import os as _os
            return _os.path.exists(path)

        def _build_path_for_n(index: int, total: int) -> str:
            """Construct the path for file ``index`` assuming ``total`` files."""
            pad_idx = _zero_pad_width(total)
            fname = (f"{file_prefix}_{str(index).zfill(pad_idx)}"
                     f"_of_{total}.{file_format}")
            root = f"{data_folder}/{dataset_type}"
            if num_subfolders > 1:
                sub_pad = _zero_pad_width(num_subfolders)
                subfolder = str(index % num_subfolders).zfill(sub_pad)
                return f"{root}/{subfolder}/{fname}"
            return f"{root}/{fname}"

        # Fast path: caller knows N — verify file 0 exists with that suffix
        if expected_total > 0:
            probe = _build_path_for_n(0, expected_total)
            if _exists(probe):
                return expected_total
            raise FileNotFoundError(
                f"SyntheticFileList.peek_total_from_disk: expected "
                f"N={expected_total} but {probe!r} does not exist. "
                f"Check that the dataset was generated with this N, or "
                f"remove dataset.num_files_train_on_disk to enable "
                f"auto-discovery.")

        # Auto-discovery fallback: scan a single directory entry to
        # parse N from a filename like "img_000_of_1000.npz".
        # This uses os.scandir and only works on local FS.
        import os as _os
        root_dir = f"{data_folder}/{dataset_type}"
        if num_subfolders > 1:
            subfolder = "0".zfill(_zero_pad_width(num_subfolders))
            search_dir = f"{root_dir}/{subfolder}"
        else:
            search_dir = root_dir
        suffix = f".{file_format}"
        try:
            scanner = _os.scandir(search_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"SyntheticFileList.peek_total_from_disk: directory "
                f"{search_dir!r} does not exist. Run with "
                f"++workload.workflow.generate_data=True or check "
                f"data_folder/dataset_type/num_subfolders.") from e
        distinct_totals: set = set()
        scanned = 0
        max_scan = 256
        with scanner as entries:
            for entry in entries:
                if scanned >= max_scan and distinct_totals:
                    break
                scanned += 1
                name = entry.name
                if not name.startswith(file_prefix) or not name.endswith(suffix):
                    continue
                stem = name[len(file_prefix) + 1:-(len(file_format) + 1)]
                marker = "_of_"
                pos = stem.rfind(marker)
                if pos < 0:
                    continue
                try:
                    distinct_totals.add(int(stem[pos + len(marker):]))
                except ValueError:
                    continue
        if not distinct_totals:
            raise FileNotFoundError(
                f"SyntheticFileList.peek_total_from_disk: no file matching "
                f"{file_prefix}_*_of_*.{file_format} found in {search_dir!r} "
                f"(scanned {scanned} entries). Either the dataset was not "
                f"generated, or file_prefix / format do not match what "
                f"datagen wrote.")
        if len(distinct_totals) == 1:
            return next(iter(distinct_totals))
        # Multiple series — ambiguous without expected_total.
        raise ValueError(
            f"SyntheticFileList.peek_total_from_disk: directory "
            f"{search_dir!r} contains files from {len(distinct_totals)} "
            f"different datagen runs (distinct '_of_N' values: "
            f"{sorted(distinct_totals)}). Pass expected_total=<N> "
            f"(in YAML: dataset.num_files_train_on_disk: <N>) to "
            f"disambiguate, or set "
            f"++workload.dataset.synthetic_file_list=False to fall back "
            f"to walking the filesystem.")

    # -- Validation -------------------------------------------------------

    def _probe_indices(self,
                       num_random_probes: int = 8,
                       seed: int = 0xC0DE) -> List[int]:
        if self._length == 0:
            return []
        candidates = {0, self._num_files_total - 1}
        if self._num_subfolders > 1:
            stride = max(1, self._num_files_total // self._num_subfolders)
            candidates.update({
                stride,
                max(0, self._num_files_total - stride - 1),
            })
        if num_random_probes > 0:
            rng = np.random.default_rng(seed)
            candidates.update(
                int(x) for x in rng.integers(
                    0, self._num_files_total, size=num_random_probes
                )
            )
        return sorted(
            i for i in candidates if 0 <= i < self._num_files_total
        )

    def validate_on_disk(self,
                         expected_record_bytes: int = 0,
                         num_random_probes: int = 8,
                         require_size_match: bool = False,
                         storage=None,
                         logger=None) -> List[Tuple[int, str, int]]:
        """Probe a handful of file paths; raise on any miss.

        Designed to be called only on rank 0.  Uses
        ``storage.get_node()`` for existence checks — works on any
        storage backend (local FS, S3, AIStore).

        Args:
            expected_record_bytes: minimum acceptable file size (only
                checked on local FS when ``require_size_match`` is set).
            num_random_probes: number of random middle indices to probe
                in addition to first/last/per-subfolder-stride.
            require_size_match: if True and storage is local FS, raise
                when any probed file is smaller than
                ``expected_record_bytes``.
            storage: a storage handler exposing ``get_node(path)``.
                When None, falls back to ``os.stat``.
            logger: optional logger; emits an INFO line on success.

        Returns:
            List of ``(file_index, path, size_bytes)`` tuples for the
            probed files.  ``size_bytes`` is 0 when using a non-local
            storage backend (get_node does not return size).
        """
        probe_results: List[Tuple[int, str, int]] = []
        for file_index in self._probe_indices(num_random_probes):
            path = self._format_path(file_index)
            if storage is not None:
                node_type = storage.get_node(path)
                if node_type is None:
                    raise FileNotFoundError(
                        f"SyntheticFileList probe failed at index "
                        f"{file_index}: expected {path!r} to exist.\n"
                        f"Possible causes: the dataset was never generated "
                        f"(run with ++workload.workflow.generate_data=True); "
                        f"the layout does not match what data_generator "
                        f"produces; or num_subfolders, file_prefix, or "
                        f"format do not match what was used at datagen time. "
                        f"As a last resort, set "
                        f"++workload.dataset.synthetic_file_list=False to "
                        f"fall back to walking the storage.")
                probe_results.append((file_index, path, 0))
            else:
                import os as _os
                try:
                    stat_result = _os.stat(path)
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"SyntheticFileList probe failed at index "
                        f"{file_index}: expected {path!r} to exist.\n"
                        f"Possible causes: the dataset was never generated "
                        f"(run with ++workload.workflow.generate_data=True); "
                        f"the layout does not match what data_generator "
                        f"produces; or num_subfolders, file_prefix, or "
                        f"format do not match what was used at datagen time. "
                        f"As a last resort, set "
                        f"++workload.dataset.synthetic_file_list=False to "
                        f"fall back to walking the filesystem.") from e
                probe_results.append((file_index, path, stat_result.st_size))
                if require_size_match and expected_record_bytes:
                    if stat_result.st_size < expected_record_bytes:
                        raise ValueError(
                            f"SyntheticFileList probe at {path!r} found "
                            f"{stat_result.st_size} B but expected >= "
                            f"{expected_record_bytes} B.")
        if logger and probe_results:
            logger.info(
                f"SyntheticFileList probed {len(probe_results)} files "
                f"(indices {probe_results[0][0]}..{probe_results[-1][0]}) "
                f"-- OK")
        return probe_results


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def shuffle_file_list(file_list, seed: int) -> None:
    """In-place shuffle that works for both legacy lists and lazy views.

    ``np.random.shuffle()`` on a plain ``list[str]`` permutes the
    references in place, which is what the legacy code did.  The lazy
    views implement an in-place ``shuffle(seed)`` on their permutation
    array instead, so swap to that path automatically when available.
    """
    if hasattr(file_list, "shuffle") and callable(file_list.shuffle):
        file_list.shuffle(seed)
    else:
        np.random.seed(int(seed))
        np.random.shuffle(file_list)

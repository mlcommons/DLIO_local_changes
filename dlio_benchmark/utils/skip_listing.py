"""
Pure helpers for the ``skip_listing`` file-name reconstruction.

When ``dataset.skip_listing`` is enabled, DLIO never lists the data folder.
Each rank rebuilds its own shard of file names from the naming convention the
data generator uses (``dlio_benchmark/data_generator/data_generator.py``):

    {file_prefix}_{index:0W}_of_{total}.{format}
    {subfolder:0S}/{file_prefix}_{index:0W}_of_{total}.{format}   (num_subfolders > 1)

where ``W = len(str(total))``, ``S = len(str(num_subfolders))`` and
``subfolder = index % num_subfolders``.

``total`` is the number of files the dataset was GENERATED with.  The number
of files a run READS (``num_files_train``) may be smaller: a submitter can
generate once and run many configurations against the first N files, exactly
as directory listing allowed (``main.py`` truncates the listed set to
``num_files_train``).  ``dataset.num_files_generated`` carries the generated
total; ``0`` means "same as ``num_files_train``", which is the behaviour before
this parameter existed.

Everything here is side-effect free so the conventions can be pinned by unit
tests without a benchmark run.
"""

import os
from typing import List


def resolve_name_total(num_files: int, num_files_generated: int) -> int:
    """The ``_of_{total}`` suffix to use for a run that reads ``num_files``.

    ``num_files_generated == 0`` (the default) keeps the pre-existing
    behaviour: the read count is the name total.  A run may never ask for
    more files than were generated — that would reconstruct names that do
    not exist, and the failure is clearer here than as a HEAD-check miss.
    """
    num_files = int(num_files)
    num_files_generated = int(num_files_generated or 0)
    if num_files_generated <= 0:
        return num_files
    if num_files > num_files_generated:
        raise ValueError(
            f"num_files_train={num_files} exceeds the generated dataset: "
            f"num_files_generated={num_files_generated}. Reduce "
            f"dataset.num_files_train or regenerate the dataset with at "
            f"least {num_files} files.")
    return num_files_generated


def file_relpath(file_prefix: str, fmt: str, idx: int, name_total: int,
                 num_subfolders: int) -> str:
    """Path of file ``idx`` relative to ``<data_folder>/<train|valid>/``.

    Mirrors ``data_generator.py`` exactly, including its two padding rules:
    the index pads to ``len(str(name_total))`` digits and the subfolder to
    ``len(str(num_subfolders))`` digits, and subfolders are only used when
    ``num_subfolders > 1``.
    """
    nd_f = len(str(name_total))
    fname = f"{file_prefix}_{str(idx).zfill(nd_f)}_of_{name_total}.{fmt}"
    if num_subfolders > 1:
        nd_sf = len(str(num_subfolders))
        return os.path.join(str(idx % num_subfolders).zfill(nd_sf), fname)
    return fname


def rank_indices(num_files: int, rank: int, comm_size: int) -> range:
    """Round-robin shard of the READ set: indices ``rank, rank+size, ...``
    below ``num_files`` (never the generated total)."""
    return range(rank, num_files, comm_size)


def validation_indices(num_files: int, interval: int) -> List[int]:
    """Indices rank 0 HEAD-checks: first, last, and every ``interval``-th
    file of the READ set.  Empty when there is nothing to read or checking
    is disabled (``interval <= 0``)."""
    if num_files <= 0 or interval <= 0:
        return []
    return sorted({0, num_files - 1} | set(range(0, num_files, interval)))

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
Configuration checks for ``storage_type: daos`` that must pass before any
DAOS call is made.

``DLIOBenchmark`` creates its storage (and therefore connects to DAOS)
before ``derive_configurations()`` and ``validate()`` run, so these checks
live in a side-effect-free function called right after ``LoadConfig``.
"""

from dlio_benchmark.common.enumerations import (
    DataLoaderType, FormatType, FrameworkType, StorageType,
)

DAOS_SUPPORTED_FORMATS = (
    FormatType.NPZ,
    FormatType.NPY,
    FormatType.JPEG,
    FormatType.PNG,
    FormatType.TFRECORD,
    FormatType.PARQUET,
)


def daos_connection_options(args):
    """Return ``(pool, container, sys_name)`` from ``storage.storage_options``.

    Raises ``Exception`` naming the missing key when ``pool`` or
    ``container`` is absent or empty.
    """
    opts = getattr(args, "storage_options", None) or {}
    pool = opts.get("pool")
    container = opts.get("container")
    if pool is None or str(pool).strip() == "":
        raise Exception(
            "storage_type=daos requires storage.storage_options.pool "
            "(the DAOS pool label or UUID).")
    if container is None or str(container).strip() == "":
        raise Exception(
            "storage_type=daos requires storage.storage_options.container "
            "(the DAOS container label or UUID; it must already exist).")
    sys_name = opts.get("sys_name")
    if sys_name is not None and str(sys_name).strip() == "":
        sys_name = None
    return str(pool), str(container), (str(sys_name) if sys_name is not None else None)


def check_daos_config(args) -> None:
    """Reject configurations that ``storage_type: daos`` does not support.

    Does nothing for other storage types.  Never connects to DAOS.
    """
    if getattr(args, "storage_type", None) != StorageType.DAOS:
        return

    daos_connection_options(args)

    if getattr(args, "data_loader", None) == DataLoaderType.NATIVE_DALI:
        raise Exception(
            "storage_type=daos does not support reader.data_loader=native_dali: "
            "DALI opens files by local path. Use the pytorch or tensorflow data loader.")

    if getattr(args, "odirect", False):
        raise Exception(
            "storage_type=daos does not support reader.odirect=True: "
            "O_DIRECT applies to local files only. Remove reader.odirect.")

    fmt = getattr(args, "format", None)
    reads = bool(getattr(args, "do_train", False) or getattr(args, "do_eval", False))
    generates = bool(getattr(args, "generate_data", False))

    if (reads or generates) and fmt not in DAOS_SUPPORTED_FORMATS:
        supported = ", ".join(f.value for f in DAOS_SUPPORTED_FORMATS)
        raise Exception(
            f"storage_type=daos does not support dataset.format={fmt}. "
            f"Supported formats: {supported}.")

    if (reads and fmt == FormatType.TFRECORD
            and getattr(args, "data_loader", None) != DataLoaderType.TENSORFLOW):
        raise Exception(
            "storage_type=daos reads dataset.format=tfrecord only with "
            "reader.data_loader=tensorflow.")

    if (generates and fmt == FormatType.PARQUET
            and getattr(args, "parquet_partition_by", None)):
        raise Exception(
            "storage_type=daos does not support dataset.parquet.partition_by: "
            "partitioned Parquet generation writes through local paths.")

    if (getattr(args, "do_checkpoint", False)
            and getattr(args, "framework", None) != FrameworkType.PYTORCH):
        raise Exception(
            f"storage_type=daos checkpointing requires framework=pytorch "
            f"(got framework={getattr(args, 'framework', None)}).")

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
``storage_type: daos`` -- a hierarchical namespace inside one DAOS POSIX
container, accessed through libdfs (no dfuse mount).

Configuration::

    storage:
      storage_type: daos
      storage_root: /            # base path inside the container
      storage_options:
        pool: <pool label or UUID>
        container: <container label or UUID>
        sys_name: <optional DAOS system name>

The namespace (``storage_root``, or ``checkpoint_folder`` for the
checkpointing storage) is a path inside the container.  ``get_uri()``
returns normalised absolute container paths; ``put_data``/``get_data``
receive such complete paths and do not prefix them again.  The pool and
container are never created here.
"""

import fnmatch
import posixpath

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.common.enumerations import MetadataType, NamespaceType
from dlio_benchmark.storage.dfs_client import get_client, normalize_path
from dlio_benchmark.storage.dfs_precheck import daos_connection_options
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

_GLOB_MAGIC = ("*", "?", "[")


def _has_magic(part):
    return any(c in part for c in _GLOB_MAGIC)


class DfsStorage(DataStorage):
    """DataStorage over a DAOS POSIX container through ``DfsClient``.

    The connection belongs to the process (``get_client``) and is shared
    with the readers and with other storage objects of the same process.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.pool, self.container, self.sys_name = daos_connection_options(self._args)
        self.namespace = Namespace(normalize_path(namespace), NamespaceType.HIERARCHICAL)

    @property
    def client(self):
        return get_client(self.pool, self.container, self.sys_name)

    @dlp.log
    def get_uri(self, id):
        if id is None or str(id) == "":
            return self.namespace.name
        return normalize_path(posixpath.join(self.namespace.name, str(id)))

    # Namespace APIs
    @dlp.log
    def create_namespace(self, exist_ok=False):
        self.client.mkdir(self.namespace.name, parents=True, exist_ok=exist_ok)
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    # Metadata APIs
    @dlp.log
    def create_node(self, id, exist_ok=False):
        self.client.mkdir(self.get_uri(id), parents=True, exist_ok=exist_ok)
        return True

    @dlp.log
    def get_node(self, id=""):
        kind, _ = self.client.stat(self.get_uri(id))
        if kind == "dir":
            return MetadataType.DIRECTORY
        if kind == "file":
            return MetadataType.FILE
        return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        """Child names of ``id``; with ``use_pattern``, matching full paths.

        A missing directory yields an empty list.
        """
        path = self.get_uri(id)
        if not use_pattern:
            return self._listdir_or_empty(path)
        last = posixpath.basename(path)
        fmt = last.split(".")[-1] if "." in last else ""
        if fmt != fmt.lower():
            raise Exception(f"Unknown file format {fmt}")
        patterns = [path]
        if fmt and fmt.upper() != fmt:
            head, _, _ = last.rpartition(".")
            patterns.append(posixpath.join(posixpath.dirname(path), f"{head}.{fmt.upper()}"))
        results = []
        seen = set()
        for pattern in patterns:
            for match in self._expand(pattern):
                if match not in seen:
                    seen.add(match)
                    results.append(match)
        return results

    def _listdir_or_empty(self, path):
        try:
            return self.client.listdir(path)
        except (FileNotFoundError, NotADirectoryError):
            return []

    def _expand(self, pattern):
        """Expand a glob pattern component by component (``glob.glob`` rules)."""
        parts = [p for p in pattern.split("/") if p]
        if not parts:
            return ["/"] if self.client.stat("/")[0] else []
        current = ["/"]
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            matched = []
            if _has_magic(part):
                for directory in current:
                    for name in self._listdir_or_empty(directory):
                        if name.startswith(".") and not part.startswith("."):
                            continue
                        if fnmatch.fnmatchcase(name, part):
                            matched.append(posixpath.join(directory, name))
            else:
                for directory in current:
                    candidate = posixpath.join(directory, part)
                    if last:
                        if self.client.stat(candidate)[0] is not None:
                            matched.append(candidate)
                    else:
                        matched.append(candidate)
            current = matched
            if not current:
                break
        return current

    @dlp.log
    def delete_node(self, id):
        self.client.remove(self.get_uri(id), recursive=True)
        return True

    # Data APIs
    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        # id is already a complete container path (callers apply get_uri()).
        self.client.write(normalize_path(id), data)
        return True

    @dlp.log
    def get_data(self, id, data=None, offset=None, length=None):
        return self.client.read(normalize_path(id), offset or 0, length)

    @dlp.log
    def isfile(self, id):
        return self.client.stat(self.get_uri(id))[0] == "file"

    def file_exists(self, id):
        return self.isfile(id)

    def get_basename(self, id):
        return posixpath.basename(str(id))

    def islocalfs(self):
        return False

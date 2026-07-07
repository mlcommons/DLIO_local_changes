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
from abc import ABC, abstractmethod
from time import time, sleep

from dlio_benchmark.common.constants import MODULE_STORAGE
from dlio_benchmark.storage.storage_handler import DataStorage, Namespace
from dlio_benchmark.common.enumerations import NamespaceType, MetadataType
import os
import glob
import shutil

from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_STORAGE)

# storage#699: os.makedirs(path, exist_ok=True) race on multi-host checkpoint
# runs over a shared/networked filesystem (NFS/GPFS/Lustre).
#
# CPython's own exist_ok handling already tolerates races between LOCAL
# creators sharing one kernel's dentry cache:
#   try: mkdir(name, mode)
#   except OSError:
#       if not exist_ok or not path.isdir(name): raise
# That is NOT reliable across HOSTS on a networked filesystem: a remote
# host's mkdir() can succeed server-side while THIS host's client is still
# serving a stale negative directory-entry lookup, so an *immediate*
# isdir() recheck — whether CPython's own or one performed microseconds
# later by a caller — can still see "not a directory" and (incorrectly)
# re-raise. This is exactly what was observed: the traceback shows
# CPython's own recheck already failed before the exception reached
# create_node(); a bare duplicate of the same immediate check would not
# have helped. Retrying the isdir() recheck with a short backoff gives a
# momentarily-stale client cache time to converge with what the other
# host's client already committed.
_MAKEDIRS_RACE_MAX_ATTEMPTS = 6
_MAKEDIRS_RACE_BASE_DELAY_S = 0.1
_MAKEDIRS_RACE_MAX_DELAY_S = 1.0


def _makedirs_race_safe(path, exist_ok):
    """``os.makedirs(path, exist_ok=exist_ok)``, hardened against concurrent
    multi-host directory creation on networked filesystems (storage#699).

    Still raises if the path is genuinely not a directory after all
    retries (e.g. a stray file at that path) — exist_ok=True must not
    become a blanket error-suppression switch.
    """
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return
    except FileExistsError:
        if not exist_ok:
            raise
        for attempt in range(_MAKEDIRS_RACE_MAX_ATTEMPTS):
            if os.path.isdir(path):
                return
            delay = min(
                _MAKEDIRS_RACE_BASE_DELAY_S * (2 ** attempt),
                _MAKEDIRS_RACE_MAX_DELAY_S,
            )
            sleep(delay)
        if not os.path.isdir(path):
            raise


class FileStorage(DataStorage):
    """
    Storage APIs for creating files.
    """

    @dlp.log_init
    def __init__(self, namespace, framework=None):
        super().__init__(framework)
        self.namespace = Namespace(namespace, NamespaceType.HIERARCHICAL)

    @dlp.log
    def get_uri(self, id):
        return os.path.join(self.namespace.name, id)

    # Namespace APIs
    @dlp.log
    def create_namespace(self, exist_ok=False):
        _makedirs_race_safe(self.namespace.name, exist_ok)
        return True

    @dlp.log
    def get_namespace(self):
        return self.namespace.name

    # Metadata APIs
    @dlp.log
    def create_node(self, id, exist_ok=False):
        _makedirs_race_safe(self.get_uri(id), exist_ok)
        return True

    @dlp.log
    def get_node(self, id=""):
        path = self.get_uri(id)
        if os.path.exists(path):
            if os.path.isdir(path):
                return MetadataType.DIRECTORY
            else:
                return MetadataType.FILE
        else:
            return None

    @dlp.log
    def walk_node(self, id, use_pattern=False):
        if not use_pattern:
            return os.listdir(self.get_uri(id))
        else:
            format= self.get_uri(id).split(".")[-1]
            upper_case = self.get_uri(id).replace(format, format.upper())
            lower_case = self.get_uri(id).replace(format, format.lower())
            if format != format.lower():
                raise Exception(f"Unknown file format {format}")
            return glob.glob(self.get_uri(id)) + glob.glob(upper_case)


    @dlp.log
    def delete_node(self, id):
        shutil.rmtree(self.get_uri(id))
        return True

    # TODO Handle partial read and writes
    @dlp.log
    def put_data(self, id, data, offset=None, length=None):
        # id is the fully-resolved path (callers call get_uri() before put_data).
        # Do NOT call self.get_uri(id) here — that would double-prefix the namespace.
        with open(id, "wb") as fd:
            fd.write(data)

    @dlp.log
    def get_data(self, id, data, offset=None, length=None):
        # id is the fully-resolved path (callers call get_uri() before put_data).
        # Do NOT call self.get_uri(id) here — that would double-prefix the namespace.
        with open(id, "rb") as fd:
            data = fd.read()
        return data
    
    @dlp.log
    def isfile(self, id):
        return os.path.isfile(id)

    def file_exists(self, id):
        """Return True if the local file exists."""
        return os.path.isfile(id)

    def get_basename(self, id):
        return os.path.basename(id)

    def islocalfs(self):
        return True

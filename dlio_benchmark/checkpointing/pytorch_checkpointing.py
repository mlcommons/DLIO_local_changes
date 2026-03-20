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
import os
import torch
import ctypes
import numpy as np
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing, get_datatype_size
from dlio_benchmark.utils.utility import Profile, dft_ai, gen_random_tensor

from dlio_benchmark.common.constants import MODULE_CHECKPOINT


class _SizePlaceholder:
    """Zero-allocation stand-in for a model tensor (file backend).

    get_tensor_core() returns this instead of a real torch.Tensor so the
    benchmark can represent 70B+ parameter models without materialising them
    in RAM.  save_state() uses StreamingCheckpointing to write the matching
    byte count via dgen-py; load_state() issues range-GETs of the same size.
    """
    __slots__ = ('size_bytes',)
    def __init__(self, num_elements: int, datatype: str = 'int8'):
        self.size_bytes = int(num_elements) * get_datatype_size(datatype)


def _compute_state_bytes(state) -> int:
    """Sum bytes of all _SizePlaceholder (or real tensor) leaves in *state*."""
    if isinstance(state, _SizePlaceholder):
        return state.size_bytes
    if isinstance(state, dict):
        return sum(_compute_state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_compute_state_bytes(v) for v in state)
    if hasattr(state, 'nbytes'):   # real torch / numpy tensor fallback
        return state.nbytes
    return 0

def get_torch_datatype(datatype):
    if datatype == "fp32":
        return torch.float32
    elif datatype == "fp16":
        return torch.float16
    elif datatype == "fp64":
        return torch.float64
    elif datatype == "int8":
        return torch.int8
    elif datatype == "uint8":
        return torch.uint8
    elif datatype == "bf16": # bfloat16
        return torch.bfloat16
    else:
        raise Exception(f"Invalid datatype {datatype}")
    

dlp = Profile(MODULE_CHECKPOINT)


class PyTorchCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchCheckpointing.__instance is None:
            PyTorchCheckpointing.__instance = PyTorchCheckpointing()
        return PyTorchCheckpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        super().__init__("pt")

    def get_tensor_core(self, length, datatype="int8", randomize=True):
        """Return a _SizePlaceholder — no tensor memory allocated."""
        return _SizePlaceholder(length, datatype)

    def _get_streaming(self):
        """Build (once per backend) a StreamingCheckpointing instance.

        Backend selection is driven by ``storage.storage_type`` in the DLIO
        config:

        * ``local_fs``  — buffered POSIX I/O + fadvise(DONTNEED) so reads
          always hit the storage device rather than the page cache.
        * ``direct_fs`` — O_DIRECT via s3dlio's ``direct://`` URI ; the kernel
          page cache is bypassed entirely, giving the cleanest possible
          measurement of raw storage throughput.  Requires s3dlio >= 0.9.x.
        """
        from dlio_benchmark.common.enumerations import StorageType

        try:
            use_direct = (self.args.storage_type == StorageType.DIRECT_FS)
        except AttributeError:
            use_direct = False

        cache_key = 'direct_fs' if use_direct else 'file'
        if not hasattr(self, '_streaming_cache'):
            self._streaming_cache = {}

        if cache_key not in self._streaming_cache:
            from mlpstorage.checkpointing import StreamingCheckpointing as _SC
            if use_direct:
                self._streaming_cache[cache_key] = _SC(
                    chunk_size=32 * 1024 * 1024,
                    num_buffers=4,
                    use_dgen=True,
                    backend='direct_fs',
                    fadvise_mode='none',   # O_DIRECT: page cache never populated
                    num_parallel_readers=4,
                )
            else:
                self._streaming_cache[cache_key] = _SC(
                    chunk_size=32 * 1024 * 1024,
                    num_buffers=4,
                    use_dgen=True,
                    backend='file',
                    fadvise_mode='dontneed',
                    num_parallel_readers=4,
                )
        return self._streaming_cache[cache_key]

    def _get_real_tensor_core(self, length, datatype="int8", randomize=True):
        """Original torch-tensor implementation (kept for unit tests / non-checkpoint use)."""
        torch_dtype=get_torch_datatype(datatype)
        if randomize:
            # Use gen_random_tensor() to leverage dgen-py (155x faster than torch.rand)
            # Maps torch dtype to numpy dtype for gen_random_tensor
            dtype_map = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.float64: np.float64,
                torch.bfloat16: np.float32,  # NumPy doesn't have bfloat16, use float32 then convert
                torch.int8: np.int8,
                torch.uint8: np.uint8,
            }
            
            if torch_dtype not in dtype_map:
                raise Exception(f"Datatype {torch_dtype} cannot be randomized for random tensor generation.")
            
            np_dtype = dtype_map[torch_dtype]
            
            # Generate data using gen_random_tensor (auto-uses dgen-py if available)
            np_array = gen_random_tensor(shape=(length,), dtype=np_dtype)
            
            # Convert to torch tensor
            tensor = torch.from_numpy(np_array)
            
            # Handle bfloat16 special case (NumPy doesn't support it)
            if torch_dtype == torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            
            return tensor
        else:
            return torch.ones(length, dtype=torch_dtype)

    def set_madvise_mergeable(self, tensor):
        """
        Apply MADV_MERGEABLE to a PyTorch tensor's memory region with alignment handling.

        1. Validates madvise is initialized and the tensor has valid memory pointers
        2. Calculates page-aligned memory boundaries for the tensor
        3. Applies madvise(MADV_MERGEABLE) to the aligned region

        Returns False immediately for _SizePlaceholder (no real memory to advise).
        """
        if not self.madvise_ready:
            return False
        if isinstance(tensor, _SizePlaceholder):
            return False

        try:
            if not (hasattr(tensor, 'data_ptr') and hasattr(tensor, 'untyped_storage')):
                 return False

            ptr_addr = tensor.data_ptr()
            storage = tensor.untyped_storage()

            if storage is None or ptr_addr == 0:
                 return False

            size_bytes = storage.nbytes()
            if size_bytes <= 0:
                return False

        except Exception:
            return False

        page_size = self.madvise_page_size
        start_addr = ptr_addr
        end_addr = ptr_addr + size_bytes

        aligned_start_addr = (start_addr + page_size - 1) // page_size * page_size
        aligned_end_addr = end_addr // page_size * page_size
        aligned_size = aligned_end_addr - aligned_start_addr

        if aligned_size <= 0:
            return False

        try:
            c_ptr = ctypes.c_void_p(aligned_start_addr)
            c_size = ctypes.c_size_t(aligned_size)
            ret = self.madvise_func(c_ptr, c_size, self.madvise_mergeable)

            if ret == 0:
                return True
            else:
                return False

        except Exception:
            return False

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync=False):
        """Stream synthetic data of the correct byte-count to the file backend.

        fsync is honoured only when the underlying OS supports it — the
        StreamingCheckpointing file writer respects it via O_DSYNC / fsync.
        """
        name        = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)
        if total_bytes <= 0:
            self.logger.warning(f"save_state: 0 bytes for '{suffix}', skipping")
            return
        self._get_streaming().save(name, total_bytes)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        """Stream-read the checkpoint file and discard data (throughput benchmark)."""
        name        = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)
        if total_bytes <= 0:
            self.logger.warning(f"load_state: 0 bytes for '{suffix}', skipping")
            return
        self._get_streaming().load(name, total_bytes)
        assert len(state.keys()) > 0

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()


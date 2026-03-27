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

import io

import h5py
import numpy as np

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor

from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in HDF5 format.
"""
class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.record_labels = [0] * self.num_samples
        self.hdf5_compression = None
        self.hdf5_compression_level = None
        if self.compression != Compression.NONE:
            self.hdf5_compression = str(self.compression)
            if self.compression == str(Compression.GZIP):
                self.hdf5_compression_level = self.compression_level

    def create_file(self, name, shape, records, **kwargs):
        hf = h5py.File(name, 'w', libver='latest')
        for dataset_id in range(self._args.num_dset_per_record):
            hf.create_dataset(f'records_{dataset_id}', shape,
                              compression=self.hdf5_compression,
                              compression_opts=self.hdf5_compression_level,
                              dtype=self._args.record_element_dtype,
                              data=records, **kwargs)
        hf.create_dataset('labels', data=self.record_labels)
        hf.close()

    @dlp.log
    def generate(self):
        """
        Generate HDF5 data for training.

        Improvements over the original:
        - Each MPI rank uses a unique seed (``BASE_SEED + my_rank``) so ranks
          produce different random data.
        - Each file uses a per-file seed (``BASE_SEED + i``) passed to
          ``gen_random_tensor``, making per-file content reproducible across runs.
        - Object-storage targets are handled via an in-memory h5py core driver.
        """
        super().generate()

        np.random.seed(self.BASE_SEED + self.my_rank)
        rng = np.random.default_rng(seed=self.BASE_SEED + self.my_rank)

        dim = self.get_dimension(self.total_files_to_generate)
        if self._args.num_dset_per_record > 1:
            dim = [[int(d[0] / self._args.num_dset_per_record), *d[1:]] for d in dim]

        kwargs = {}
        if len(self._args.chunk_dims) > 0:
            kwargs["chunks"] = self._args.chunk_dims

        is_local = self.storage.islocalfs()

        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1 = dim[2 * i]
            if isinstance(dim1, list):
                if dim1[0] == 1:
                    dim1 = dim1[1:]
                shape = (self.num_samples, *dim1) if self.num_samples > 1 else (1, *dim1)
                if len(self._args.max_shape) > 0:
                    kwargs["maxshape"] = (shape[0], *self._args.max_shape)
                records = gen_random_tensor(shape=shape, dtype=self._args.record_element_dtype, rng=rng)
            else:
                dim2 = dim[2 * i + 1]
                shape = (self.num_samples, dim1, dim2) if self.num_samples > 1 else (1, dim1, dim2)
                if len(self._args.max_shape) > 0:
                    kwargs["maxshape"] = (shape[0], *self._args.max_shape)
                records = gen_random_tensor(shape=shape, dtype=self._args.record_element_dtype, rng=rng)

            progress(i + 1, self.total_files_to_generate, "Generating HDF5 Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])

            if is_local:
                self.create_file(name=out_path_spec, shape=shape, records=records, **kwargs)
            else:
                # Use h5py in-memory core driver for object-storage targets.
                hf = h5py.File('in-memory.h5', 'w', driver='core',
                               backing_store=False, libver='latest')
                for dataset_id in range(self._args.num_dset_per_record):
                    hf.create_dataset(f'records_{dataset_id}', shape,
                                      compression=self.hdf5_compression,
                                      compression_opts=self.hdf5_compression_level,
                                      dtype=self._args.record_element_dtype,
                                      data=records, **kwargs)
                hf.create_dataset('labels', data=self.record_labels)
                hf.flush()
                self.storage.put_data(out_path_spec, bytes(hf.id.get_file_image()))
                hf.close()

        np.random.seed()

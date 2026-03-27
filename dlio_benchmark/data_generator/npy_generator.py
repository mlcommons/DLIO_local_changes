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
import numpy as np

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in NPY format.
"""
class NPYGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPY format of 3d dataset.
        Uses the base-class template for seeding, BytesIO, and put_data.
        """
        super().generate()
        dtype = self._args.record_element_dtype
        num_samples = self.num_samples

        def _write(i, dim_, dim1, dim2, file_seed, rng,
                   out_path_spec, is_local, output):
            if isinstance(dim_, list):
                shape = (*dim_, num_samples)
            else:
                shape = (dim1, dim2, num_samples)
            records = gen_random_tensor(shape=shape, dtype=dtype, rng=rng)
            np.save(output, records)

        self._generate_files(_write, "NPY Data")

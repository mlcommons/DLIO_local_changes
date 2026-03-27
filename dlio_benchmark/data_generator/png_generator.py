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
import PIL.Image as im

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, utcnow, gen_random_tensor
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

class PNGGenerator(DataGenerator):
    @dlp.log
    def generate(self):
        """
        Generator for creating data in PNG format of 3d dataset.
        Uses the base-class template for seeding, BytesIO, and put_data.
        """
        super().generate()
        my_rank = self.my_rank
        total = self.total_files_to_generate
        logger = self.logger

        def _write(i, dim_, dim1, dim2, file_seed, rng,
                   out_path_spec, is_local, output):
            records = gen_random_tensor(shape=(dim1, dim2), dtype=np.uint8,
                                        rng=rng)
            records = np.clip(records, 0, 255).astype(np.uint8)
            if my_rank == 0:
                logger.debug(f"{utcnow()} Dimension of images: {dim1} x {dim2}")
            img = im.fromarray(records)
            if my_rank == 0 and i % 100 == 0:
                logger.info(f"Generated file {i}/{total}")
            img.save(output, format='PNG')

        self._generate_files(_write, "PNG Data")

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
import numpy as np

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

class SyntheticGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        """
        Generator for creating dummy files.

        Each file contains its global file index i as a UTF-8 string.
        Uses the _generate_files template so that:
        - Rank-unique seeds are set before the loop.
        - Object storage is handled transparently (BytesIO path).
        - Local filesystem writes go directly to the resolved path.
        """
        super().generate()

        def _write(i, dim_, dim1, dim2, file_seed, rng, out_path_spec, is_local, output):
            if self.my_rank == 0 and i % 100 == 0:
                self.logger.info(f"Generated file {i}/{self.total_files_to_generate}")
            content = f"{i}".encode("utf-8")
            if is_local:
                with open(out_path_spec, "wb") as f:
                    f.write(content)
            else:
                output.write(content)

        self._generate_files(_write, "Synthetic Data (Empty)")

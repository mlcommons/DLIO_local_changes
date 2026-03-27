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
import pandas as pd

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress, gen_random_tensor

"""
Generator for creating data in CSV format.
"""
class CSVGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generate csv data for training. It generates a 2d dataset and writes it to file.
        Supports both local filesystem and object storage targets via BytesIO serialization.
        """
        super().generate()
        dtype = self._args.record_element_dtype
        num_samples = self.num_samples
        compression_type = self.compression

        def _write(i, dim_, dim1, dim2, file_seed, rng,
                   out_path_spec, is_local, output):
            if isinstance(dim_, list):
                shape = dim_
            else:
                shape = (dim1, dim2)
            total_size = int(np.prod(shape))
            # Generate unique data for ALL samples at once with a single call.
            # Formerly this generated ONE record and tiled it num_samples times,
            # which made every row in every CSV file identical — a correctness bug.
            # Now each row (sample) gets a distinct slice of the dgen/RNG stream.
            records = gen_random_tensor(shape=(num_samples, total_size), dtype=dtype, rng=rng)
            df = pd.DataFrame(data=records)

            compression = None
            local_path = out_path_spec
            if compression_type != Compression.NONE:
                compression = {"method": str(compression_type)}
                if is_local:
                    if compression_type == Compression.GZIP:
                        local_path = out_path_spec + ".gz"
                    elif compression_type == Compression.BZIP2:
                        local_path = out_path_spec + ".bz2"
                    elif compression_type == Compression.ZIP:
                        local_path = out_path_spec + ".zip"
                    elif compression_type == Compression.XZ:
                        local_path = out_path_spec + ".xz"

            if is_local:
                df.to_csv(local_path, compression=compression,
                          index=False, header=False)
            else:
                buf = io.StringIO()
                df.to_csv(buf, compression=None, index=False, header=False)
                output.write(buf.getvalue().encode('utf-8'))

        self._generate_files(_write, "CSV Data")

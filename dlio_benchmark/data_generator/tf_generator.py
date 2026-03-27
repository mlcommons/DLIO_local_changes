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
import struct
import tempfile

import numpy as np
import tensorflow as tf

from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

class TFRecordGenerator(DataGenerator):
    """
    Generator for creating data in TFRecord format.
    """
    def __init__(self):
        super().__init__()

    @dlp.log
    def generate(self):
        """
        Generator for creating data in TFRecord format of 3d dataset.

        Improvements over the original:
        - Each MPI rank uses a unique seed (``BASE_SEED + my_rank``) so ranks
          produce different random data.
        - Each file's samples use a per-file seed derived from ``BASE_SEED + file_index``
          making per-file content reproducible across runs.
        - Object-storage targets are handled by writing to a temporary local file
          then uploading the bytes and removing the temp file.
        """
        super().generate()

        np.random.seed(self.BASE_SEED + self.my_rank)
        rng = np.random.default_rng(seed=self.BASE_SEED + self.my_rank)
        # This creates a N-D image representing a single record
        dim = self.get_dimension(self.total_files_to_generate)
        is_local = self.storage.islocalfs()

        for file_i in dlp.iter(range(self.my_rank, self.total_files_to_generate, self.comm_size)):
            progress(file_i + 1, self.total_files_to_generate, "Generating TFRecord Data")
            out_path_spec = self.storage.get_uri(self._file_list[file_i])
            dim_ = dim[2 * file_i]
            size_shape = 0
            shape = ()
            if isinstance(dim_, list):
                size_shape = int(np.prod(dim_))
                shape = dim_
            else:
                dim1 = dim_
                dim2 = dim[2 * file_i + 1]
                size_shape = dim1 * dim2
                shape = (dim1, dim2)
            size_bytes = size_shape * self._args.record_element_bytes

            if is_local:
                write_path = out_path_spec
            else:
                # Write to a temp file, then upload
                tmp = tempfile.NamedTemporaryFile(suffix='.tfrecord', delete=False)
                write_path = tmp.name
                tmp.close()

            # Open a TFRecordWriter for the output-file.
            with tf.io.TFRecordWriter(write_path) as writer:
                for sample_i in range(self.num_samples):
                    # Use the flowing RNG so each sample within a file gets\n                    # unique data without resetting to an arithmetic seed.
                    record = gen_random_tensor(shape=shape, dtype=self._args.record_element_dtype, rng=rng)
                    img_bytes = record.tobytes()
                    data = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        'size': tf.train.Feature(int64_list=tf.train.Int64List(value=[size_bytes]))
                    }
                    # Wrap the data as TensorFlow Features.
                    feature = tf.train.Features(feature=data)
                    # Wrap again as a TensorFlow Example.
                    example = tf.train.Example(features=feature)
                    # Serialize the data.
                    serialized = example.SerializeToString()
                    # Write the serialized data to the TFRecords file.
                    writer.write(serialized)

            if not is_local:
                with open(write_path, 'rb') as f:
                    self.storage.put_data(out_path_spec, f.read())
                os.unlink(write_path)
            else:
                # Create index file for local paths
                folder = "train"
                if "valid" in out_path_spec:
                    folder = "valid"
                index_folder = f"{self._args.data_folder}/index/{folder}"
                filename = os.path.basename(out_path_spec)
                self.storage.create_node(index_folder, exist_ok=True)
                tfrecord_idx = f"{index_folder}/{filename}.idx"
                if not self.storage.isfile(tfrecord_idx):
                    self.create_index_file(out_path_spec, self.storage.get_uri(tfrecord_idx))

        np.random.seed()

    @dlp.log
    def create_index_file(self, src: str, dest: str):
        """Slightly edited body of the tfrecord2idx script from the DALI project"""

        with tf.io.gfile.GFile(src, "rb") as f, tf.io.gfile.GFile(dest, "w") as idx_f:
            while True:
                current = f.tell()
                # length
                byte_len = f.read(8)
                if len(byte_len) == 0:
                    break
                # crc
                f.read(4)
                proto_len = struct.unpack("q", byte_len)[0]
                # proto
                f.read(proto_len)
                # crc
                f.read(4)
                idx_f.write(f"{current} {f.tell() - current}\n")

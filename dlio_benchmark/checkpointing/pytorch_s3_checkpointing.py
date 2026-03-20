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
from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.checkpointing.pytorch_checkpointing import (
    PyTorchCheckpointing,
    _SizePlaceholder,
    _compute_state_bytes,
)
from dlio_benchmark.utils.utility import Profile, dft_ai

from dlio_benchmark.common.constants import MODULE_CHECKPOINT
from s3torchconnector import S3Checkpoint, S3ClientConfig

dlp = Profile(MODULE_CHECKPOINT)

class PyTorchS3Checkpointing(PyTorchCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyTorchS3Checkpointing.__instance is None:
            PyTorchS3Checkpointing.__instance = PyTorchS3Checkpointing()
        return PyTorchS3Checkpointing.__instance

    @dft_ai.checkpoint.init
    def __init__(self):
        BaseCheckpointing.__init__(self, "pts3")

        # Access config values from self.args (inherited from BaseCheckpointing)
        storage_options = getattr(self.args, "storage_options", {}) or {}

        self.access_key_id = storage_options.get("access_key_id")
        self.secret_access_key = storage_options.get("secret_access_key")
        self.endpoint = storage_options.get("endpoint_url")
        self.region = storage_options.get("region", self.args.s3_region)

        if self.access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key

        # Build connector config, possibly with config overrides
        force_path_style_opt = self.args.s3_force_path_style
        if "s3_force_path_style" in storage_options:
            force_path_style_opt = storage_options["s3_force_path_style"].strip().lower() == "true"
        max_attempts_opt = self.args.s3_max_attempts
        if "s3_max_attempts" in storage_options:
            try:
                max_attempts_opt = int(storage_options["s3_max_attempts"])
            except (TypeError, ValueError):
                max_attempts_opt = self.args.s3_max_attempt
        self.s3_client_config = S3ClientConfig(
            force_path_style=force_path_style_opt,
            max_attempts=max_attempts_opt,
        )

        # Initialize the S3Checkpoint instance
        self.s3_checkpoint = S3Checkpoint(
            region=self.region,
            endpoint=self.endpoint,
            s3client_config=self.s3_client_config,
        )

    @dft_ai.checkpoint.capture
    def save_state(self, suffix, state, fsync=False):
        """Stream synthetic data of the correct byte-count via s3torchconnector."""
        name        = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)
        if total_bytes <= 0:
            return
        self._get_streaming().save(name, total_bytes)

    @dft_ai.checkpoint.restart
    def load_state(self, suffix, state):
        """Stream-read checkpoint via s3torchconnector and discard data."""
        name        = self.get_name(suffix)
        total_bytes = _compute_state_bytes(state)
        if total_bytes <= 0:
            return
        self._get_streaming().load(name, total_bytes)
        assert len(state.keys()) > 0

    def _get_streaming(self):
        """Build (once) a StreamingCheckpointing for the s3torchconnector backend."""
        if not hasattr(self, '_streaming'):
            from mlpstorage.checkpointing import StreamingCheckpointing as _SC
            self._streaming = _SC(
                chunk_size=32 * 1024 * 1024,
                num_buffers=4,
                use_dgen=True,
                backend='s3torchconnector',
                num_parallel_readers=8,
            )
        return self._streaming

    def get_tensor_core(self, length, datatype="int8", randomize=True):
        """Return a _SizePlaceholder \u2014 no tensor memory allocated."""
        return _SizePlaceholder(length, datatype)

    @dlp.log
    def save_checkpoint(self, epoch, step_number):
        super().save_checkpoint(epoch, step_number)

    @dlp.log
    def load_checkpoint(self, epoch, step_number):
        super().load_checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()


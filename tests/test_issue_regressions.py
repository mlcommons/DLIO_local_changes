"""
TDD tests for remaining DLIO benchmark issues.

Issues covered (PRs #12–#15):
  - Issue 12: MPI topology used for thread auto-sizing  (PR-13)
  - Issues 10+11+6b: Parallel data generation           (PR-14)
  - Issue 9: Storage env-var overrides                  (PR-12)
  - Issue 13: Post-generation settle guard              (PR-15)

Workflow:
    Write test → run (must FAIL) → implement fix → run (must PASS)
"""

import hashlib
import io
import os
import shutil
import tempfile
import time
from unittest import mock
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DLIO_TEST_OUTPUT_DIR = os.environ.get("DLIO_TEST_OUTPUT_DIR", "dlio_test_output")
os.environ.setdefault("DLIO_OUTPUT_FOLDER", DLIO_TEST_OUTPUT_DIR)

import dlio_benchmark
_CONFIG_DIR = os.path.dirname(dlio_benchmark.__file__) + "/configs/"

_BASE_OVERRIDES = [
    "++workload.framework=tensorflow",
    "++workload.reader.data_loader=tensorflow",
    "++workload.workflow.train=False",
    "++workload.dataset.num_samples_per_file=4",
    "++workload.dataset.record_length=256",
]


def _reset_singletons():
    """Clear all DLIO singletons between tests."""
    from dlio_benchmark.utils.utility import DLIOMPI
    from dlio_benchmark.utils.config import ConfigArguments
    import dlio_benchmark.utils.utility as _util
    DLIOMPI.reset()
    ConfigArguments.reset()
    # Reset the process-level dgen-py singleton so each test starts fresh.
    _util._DGEN_PROC_GEN = None
    _util._DGEN_PROC_GEN_CAPACITY = 0


def _init_mpi():
    from dlio_benchmark.utils.utility import DLIOMPI
    inst = DLIOMPI.get_instance()
    inst.initialize()
    return inst


def _make_benchmark(extra_overrides=(), tmpdir=None):
    """Create a DLIOBenchmark via Hydra compose using the standard pattern."""
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.main import DLIOBenchmark

    overrides = list(_BASE_OVERRIDES) + list(extra_overrides)
    if tmpdir:
        overrides += [
            f"++workload.dataset.data_folder={tmpdir}/data",
            f"++workload.output.folder={tmpdir}/output",
        ]

    ConfigArguments.reset()
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    workload_dict = OmegaConf.to_container(cfg["workload"], resolve=True)
    if tmpdir:
        workload_dict.setdefault("output", {})["folder"] = f"{tmpdir}/output"
    return DLIOBenchmark(workload_dict)


# ===========================================================================
# Issue 12 — MPI topology for thread auto-sizing (PR-13)
# ===========================================================================


class TestIssue12_RanksPerNode:
    """DLIOMPI must expose ranks_per_node() that is safe in all MPI states."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_ranks_per_node_method_exists(self):
        """DLIOMPI must have a ranks_per_node() method."""
        from dlio_benchmark.utils.utility import DLIOMPI
        inst = DLIOMPI.get_instance()
        assert hasattr(inst, "ranks_per_node"), (
            "DLIOMPI must have a ranks_per_node() method (Issue 12 not implemented)"
        )

    def test_ranks_per_node_safe_in_child_state(self):
        """ranks_per_node() must not raise AttributeError in CHILD_INITIALIZED state."""
        from dlio_benchmark.utils.utility import DLIOMPI
        inst = DLIOMPI.get_instance()
        inst.set_parent_values(parent_rank=0, parent_comm_size=32)
        # Must not raise — child processes lack full topology info
        rpn = inst.ranks_per_node()
        assert rpn >= 1, "ranks_per_node() must return ≥ 1 in child state"

    def test_ranks_per_node_matches_npernode_after_mpi_init(self):
        """ranks_per_node() == npernode() after full MPI initialization."""
        inst = _init_mpi()
        assert inst.ranks_per_node() == inst.npernode()

    def test_ranks_per_node_positive_after_mpi_init(self):
        """ranks_per_node() must be ≥ 1 and ≤ comm_size after MPI init."""
        inst = _init_mpi()
        rpn = inst.ranks_per_node()
        assert 1 <= rpn <= inst.size()


class TestIssue12_AutoSizingDenominator:
    """read_threads auto-sizing must use ranks_per_node as denominator."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_read_threads_auto_sizing_calls_ranks_per_node(self):
        """derive_configurations() must call ranks_per_node() for thread sizing."""
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.utils.config import ConfigArguments

        mpi = _init_mpi()
        args = ConfigArguments.get_instance()
        # Ensure read_threads is at the auto-size sentinel
        args.read_threads = 1

        with patch.object(mpi, "ranks_per_node", wraps=mpi.ranks_per_node) as mock_rpn:
            args.derive_configurations()
            # ranks_per_node() must have been called during auto-sizing
            assert mock_rpn.call_count >= 1, (
                "derive_configurations() must call ranks_per_node() for thread auto-sizing "
                "(Issue 12 not implemented: still using comm_size)"
            )


# ===========================================================================
# Issue 10+11+6b — Parallel data generation (PR-14)
# ===========================================================================


class TestIssue10_WriteThreadsField:
    """ConfigArguments must expose a write_threads field."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def test_write_threads_field_exists(self):
        """ConfigArguments dataclass must have a write_threads field."""
        from dlio_benchmark.utils.config import ConfigArguments
        _init_mpi()
        args = ConfigArguments.get_instance()
        assert hasattr(args, "write_threads"), (
            "ConfigArguments must have write_threads field (Issue 10 not implemented)"
        )

    def test_write_threads_default_is_one(self):
        """write_threads default must be 1 (auto-size sentinel)."""
        from dlio_benchmark.utils.config import ConfigArguments
        _init_mpi()
        args = ConfigArguments.get_instance()
        assert args.write_threads == 1, (
            "write_threads default must be 1 (auto-size sentinel)"
        )

    def test_write_threads_auto_sized_after_derive(self):
        """After derive_configurations(), write_threads must be >= 1."""
        from dlio_benchmark.utils.config import ConfigArguments

        _init_mpi()
        args = ConfigArguments.get_instance()
        args.derive_configurations()

        # After auto-sizing, write_threads must be ≥ 1
        assert args.write_threads >= 1, "write_threads must be ≥ 1 after derive_configurations()"


class TestIssue10_ParallelGeneration:
    """_generate_files() must use ThreadPoolExecutor when write_threads > 1."""

    def setup_method(self):
        _reset_singletons()
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _reset_singletons()

    def _run_generation(self, write_threads, num_files=8):
        """Run just the data generation phase; returns the DLIOBenchmark instance."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.main import DLIOBenchmark
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf

        _init_mpi()
        ConfigArguments.reset()
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(
                config_name="config",
                overrides=_BASE_OVERRIDES + [
                    "++workload.workflow.generate_data=True",
                    f"++workload.dataset.format=npy",
                    f"++workload.dataset.num_files_train={num_files}",
                    "++workload.dataset.num_files_eval=0",
                    "++workload.dataset.num_subfolders_train=0",
                    "++workload.dataset.num_subfolders_eval=0",
                    f"++workload.dataset.data_folder={self.tmpdir}/data",
                    f"++workload.output.folder={self.tmpdir}/output",
                    f"++workload.reader.write_threads={write_threads}",
                ],
            )
        workload_dict = OmegaConf.to_container(cfg["workload"], resolve=True)
        workload_dict.setdefault("output", {})["folder"] = f"{self.tmpdir}/output"
        bench = DLIOBenchmark(workload_dict)
        bench.initialize()  # runs generation + file walk
        return bench

    def _file_hashes(self, train_dir):
        hashes = {}
        for fname in sorted(os.listdir(train_dir)):
            if fname.endswith(".npy"):
                path = os.path.join(train_dir, fname)
                with open(path, "rb") as fh:
                    hashes[fname] = hashlib.md5(fh.read()).hexdigest()
        return hashes

    def test_thread_pool_invoked_when_write_threads_gt_1(self):
        """_generate_files() must use ThreadPoolExecutor when write_threads > 1."""
        from concurrent.futures import ThreadPoolExecutor

        with patch("dlio_benchmark.data_generator.data_generator.ThreadPoolExecutor",
                   wraps=ThreadPoolExecutor) as mock_tpe:
            self._run_generation(write_threads=4, num_files=8)
            assert mock_tpe.called, (
                "_generate_files() must use ThreadPoolExecutor when write_threads > 1 "
                "(Issue 10 not implemented: still serial)"
            )

    def test_all_files_created_parallel(self):
        """Parallel generation must create all expected files."""
        self._run_generation(write_threads=4, num_files=8)

        train_dir = os.path.join(self.tmpdir, "data", "train")
        created_files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]
        assert len(created_files) == 8, (
            f"Expected 8 .npy files, got {len(created_files)}"
        )

    def test_determinism_parallel_equals_serial(self):
        """Parallel generation (write_threads=4) must produce the same number of files
        as serial (write_threads=1).  Bit-for-bit reproducibility is explicitly NOT
        required — dgen-py produces non-deterministic data by design (benchmarks
        care only about throughput, not content)."""
        # Serial run
        self._run_generation(write_threads=1, num_files=4)
        train_dir = os.path.join(self.tmpdir, "data", "train")
        serial_count = len([f for f in os.listdir(train_dir) if f.endswith(".npy")])

        # Clear and re-generate with parallel
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self.tmpdir)
        _reset_singletons()

        self._run_generation(write_threads=4, num_files=4)
        parallel_count = len([f for f in os.listdir(train_dir) if f.endswith(".npy")])

        assert serial_count == parallel_count == 4, (
            f"Expected 4 files for both serial and parallel runs, "
            f"got serial={serial_count}, parallel={parallel_count}"
        )

    def test_issue6b_comment_in_init(self):
        """DataGenerator.__init__ must contain a comment clarifying the 6b non-issue."""
        import inspect
        from dlio_benchmark.data_generator.data_generator import DataGenerator
        src = inspect.getsource(DataGenerator.__init__)
        assert "derive_configurations" in src and ("validate" in src or "6b" in src), (
            "DataGenerator.__init__() must contain a clarifying comment about "
            "derive_configurations() vs validate() (Issue 6b)"
        )


# ===========================================================================
# Issue 9 — Storage env-var overrides (PR-12)
# ===========================================================================


class TestIssue9_StorageEnvOverrides:
    """_apply_env_overrides() must populate storage_options from env vars."""

    def setup_method(self):
        _reset_singletons()
        # Remove any storage-related env vars that might bleed between tests
        for key in [
            "DLIO_STORAGE_LIBRARY", "DLIO_BUCKET", "DLIO_STORAGE_TYPE",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL", "AWS_REGION",
        ]:
            os.environ.pop(key, None)

    def teardown_method(self):
        _reset_singletons()
        for key in [
            "DLIO_STORAGE_LIBRARY", "DLIO_BUCKET", "DLIO_STORAGE_TYPE",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL", "AWS_REGION",
        ]:
            os.environ.pop(key, None)

    def _make_args(self):
        """Return a fresh ConfigArguments instance after MPI init."""
        from dlio_benchmark.utils.config import ConfigArguments
        _init_mpi()
        return ConfigArguments.get_instance()

    def _apply(self, args, dotenv=None):
        """Call _apply_env_overrides with given args and dotenv dict."""
        from dlio_benchmark.utils.config import _apply_env_overrides
        _apply_env_overrides(args, dotenv or {})

    # ── existence check ──────────────────────────────────────────────────

    def test_dlio_storage_library_env_var_applied(self):
        """DLIO_STORAGE_LIBRARY env var must set storage_options['storage_library']."""
        os.environ["DLIO_STORAGE_LIBRARY"] = "s3dlio"
        args = self._make_args()
        args.storage_options = None  # not set by YAML
        self._apply(args)
        assert args.storage_options is not None, "_apply_env_overrides must create storage_options dict"
        assert args.storage_options.get("storage_library") == "s3dlio", (
            "storage_options['storage_library'] must be set from DLIO_STORAGE_LIBRARY env var "
            "(Issue 9 not implemented)"
        )

    def test_aws_access_key_applied_to_storage_options(self):
        """AWS_ACCESS_KEY_ID env var must write into storage_options."""
        os.environ["AWS_ACCESS_KEY_ID"] = "testkey"
        args = self._make_args()
        args.storage_options = None
        self._apply(args)
        assert args.storage_options is not None
        assert args.storage_options.get("access_key_id") == "testkey", (
            "storage_options['access_key_id'] must be set from AWS_ACCESS_KEY_ID "
            "(Issue 9 not implemented)"
        )

    def test_aws_secret_key_applied_to_storage_options(self):
        """AWS_SECRET_ACCESS_KEY env var must write into storage_options."""
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testsecret"
        args = self._make_args()
        args.storage_options = None
        self._apply(args)
        assert args.storage_options.get("secret_access_key") == "testsecret"

    def test_aws_endpoint_url_applied_to_storage_options(self):
        """AWS_ENDPOINT_URL env var must write into storage_options."""
        os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
        args = self._make_args()
        args.storage_options = None
        self._apply(args)
        assert args.storage_options.get("endpoint_url") == "http://localhost:9000"

    def test_aws_region_applied_to_storage_options(self):
        """AWS_REGION env var must write into storage_options."""
        os.environ["AWS_REGION"] = "us-west-2"
        args = self._make_args()
        args.storage_options = None
        self._apply(args)
        assert args.storage_options.get("region") == "us-west-2", (
            "storage_options['region'] must be set from AWS_REGION env var "
            "(Issue 9 not implemented)"
        )

    def test_dlio_bucket_applied_to_storage_root(self):
        """DLIO_BUCKET env var must set storage_root when not already set."""
        os.environ["DLIO_BUCKET"] = "my-bucket"
        args = self._make_args()
        args.storage_root = None  # unset
        self._apply(args)
        assert args.storage_root == "my-bucket", (
            "storage_root must be set from DLIO_BUCKET env var "
            "(Issue 9 not implemented)"
        )

    def test_dlio_storage_type_applied(self):
        """DLIO_STORAGE_TYPE env var must set storage_type when not already set."""
        from dlio_benchmark.common.enumerations import StorageType
        os.environ["DLIO_STORAGE_TYPE"] = "s3"
        args = self._make_args()
        args.storage_type = None  # unset
        self._apply(args)
        assert args.storage_type == StorageType.S3, (
            "storage_type must be set from DLIO_STORAGE_TYPE env var "
            "(Issue 9 not implemented)"
        )

    # ── precedence: YAML/CLI values must not be overwritten ──────────────

    def test_yaml_storage_library_not_overwritten_by_env(self):
        """An existing storage_options value must NOT be overwritten by env var."""
        os.environ["DLIO_STORAGE_LIBRARY"] = "minio"
        args = self._make_args()
        args.storage_options = {"storage_library": "s3dlio"}  # set by YAML
        self._apply(args)
        assert args.storage_options["storage_library"] == "s3dlio", (
            "Env var must not overwrite existing YAML/CLI storage_options values"
        )

    def test_yaml_storage_root_not_overwritten_by_env(self):
        """An existing storage_root must NOT be overwritten by DLIO_BUCKET."""
        os.environ["DLIO_BUCKET"] = "env-bucket"
        args = self._make_args()
        args.storage_root = "yaml-bucket"  # set by YAML
        self._apply(args)
        assert args.storage_root == "yaml-bucket"

    # ── dotenv file support ──────────────────────────────────────────────

    def test_dotenv_file_sets_storage_options(self):
        """Values from .env file dict must set storage_options when env var absent."""
        args = self._make_args()
        args.storage_options = None
        dotenv = {"DLIO_STORAGE_LIBRARY": "s3dlio", "AWS_REGION": "eu-west-1"}
        self._apply(args, dotenv)
        assert args.storage_options is not None
        assert args.storage_options.get("storage_library") == "s3dlio", (
            ".env file values must populate storage_options via _apply_env_overrides "
            "(Issue 9 not implemented)"
        )
        assert args.storage_options.get("region") == "eu-west-1"

    def test_env_var_takes_priority_over_dotenv(self):
        """Shell env var must take priority over .env file value."""
        os.environ["DLIO_STORAGE_LIBRARY"] = "minio"
        args = self._make_args()
        args.storage_options = None
        dotenv = {"DLIO_STORAGE_LIBRARY": "s3dlio"}  # lower priority
        self._apply(args, dotenv)
        assert args.storage_options["storage_library"] == "minio", (
            "Shell env var must override .env file value"
        )


# ===========================================================================
# Issue 13 — Post-generation settle guard (PR-15)
# ===========================================================================


class TestIssue13_SettleGuard:
    """A post-generation settle time must be observed for non-local storage."""

    def setup_method(self):
        _reset_singletons()

    def teardown_method(self):
        _reset_singletons()

    def _make_args(self):
        from dlio_benchmark.utils.config import ConfigArguments
        _init_mpi()
        return ConfigArguments.get_instance()

    def test_post_generation_settle_seconds_field_exists(self):
        """ConfigArguments must have post_generation_settle_seconds field."""
        args = self._make_args()
        assert hasattr(args, "post_generation_settle_seconds"), (
            "ConfigArguments must have post_generation_settle_seconds field "
            "(Issue 13 not implemented)"
        )

    def test_post_generation_settle_seconds_default_zero(self):
        """post_generation_settle_seconds default must be 0.0 (no behavior change)."""
        args = self._make_args()
        assert args.post_generation_settle_seconds == 0.0, (
            "post_generation_settle_seconds default must be 0.0"
        )

    def test_settle_sleep_called_for_s3_with_positive_value(self):
        """time.sleep must be called when storage_type=S3 and settle > 0."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import StorageType
        from mpi4py import MPI

        mpi = _init_mpi()
        args = ConfigArguments.get_instance()
        args.storage_type = StorageType.S3
        args.post_generation_settle_seconds = 0.05
        args.my_rank = 0

        from dlio_benchmark.main import _apply_settle_guard
        with patch("dlio_benchmark.main.time") as mock_time:
            mock_time.sleep = MagicMock()
            _apply_settle_guard(args, MPI.COMM_WORLD)
            assert mock_time.sleep.called, (
                "time.sleep must be called when storage_type=S3 and "
                "post_generation_settle_seconds > 0 (Issue 13 not implemented)"
            )
            assert mock_time.sleep.call_args[0][0] == pytest.approx(0.05, rel=0.01)

    def test_no_sleep_for_local_fs(self):
        """time.sleep must NOT be called when storage_type=LOCAL_FS."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import StorageType
        from mpi4py import MPI

        _init_mpi()
        args = ConfigArguments.get_instance()
        args.storage_type = StorageType.LOCAL_FS
        args.post_generation_settle_seconds = 5.0
        args.my_rank = 0

        from dlio_benchmark.main import _apply_settle_guard
        with patch("dlio_benchmark.main.time") as mock_time:
            mock_time.sleep = MagicMock()
            _apply_settle_guard(args, MPI.COMM_WORLD)
            assert not mock_time.sleep.called, (
                "time.sleep must NOT be called when storage_type=LOCAL_FS"
            )

    def test_no_sleep_when_settle_is_zero(self):
        """time.sleep must NOT be called when post_generation_settle_seconds=0."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import StorageType
        from mpi4py import MPI

        _init_mpi()
        args = ConfigArguments.get_instance()
        args.storage_type = StorageType.S3
        args.post_generation_settle_seconds = 0.0
        args.my_rank = 0

        from dlio_benchmark.main import _apply_settle_guard
        with patch("dlio_benchmark.main.time") as mock_time:
            mock_time.sleep = MagicMock()
            _apply_settle_guard(args, MPI.COMM_WORLD)
            assert not mock_time.sleep.called, (
                "time.sleep must NOT be called when post_generation_settle_seconds=0.0"
            )


# ===========================================================================
# storage#699 — os.makedirs(exist_ok=True) race on multi-host checkpoint runs
# ===========================================================================
#
# FileStorage.create_node()/create_namespace() call os.makedirs(path,
# exist_ok=True). CPython's own exist_ok handling already tolerates races
# between LOCAL creators sharing one kernel's dentry cache (a FileExistsError
# -> os.path.isdir() recheck, see cpython os.makedirs). It is NOT reliable
# across HOSTS on a shared/networked filesystem: a remote host's mkdir() can
# succeed server-side while THIS host's client still serves a stale negative
# directory-entry lookup for a brief window, so an *immediate* isdir()
# recheck can still see "not a directory" and incorrectly re-raise. This
# surfaced deterministically on an 8-host/64-rank llama3-70b checkpoint run
# writing to a shared filesystem (storage#699) -- the traceback shows
# CPython's own recheck had already failed, so simply repeating the same
# immediate check would not help; the fix retries with backoff so a
# momentarily-stale client cache has time to converge.


class TestStorage699_MakedirsRaceOnSharedFS:
    """Regression: create_namespace()/create_node() must tolerate a
    FileExistsError from a concurrent creator whose write has not yet become
    visible to this process's os.path.isdir() check (storage#699)."""

    # -- the core fix: retry survives a momentarily-stale isdir() ------------

    def test_survives_stale_cache_then_converges(self):
        """isdir() is False on the first two rechecks (stale client cache),
        then True (cache caught up) -- must NOT raise, and must have
        actually slept between attempts (proving the retry path executed,
        not that it got lucky on attempt 1)."""
        from dlio_benchmark.storage.file_storage import _makedirs_race_safe

        with patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=FileExistsError(17, "File exists")), \
             patch("dlio_benchmark.storage.file_storage.os.path.isdir",
                   side_effect=[False, False, True]) as mock_isdir, \
             patch("dlio_benchmark.storage.file_storage.sleep") as mock_sleep:
            _makedirs_race_safe("/shared/ckpt/global_epoch1_step2", exist_ok=True)

        assert mock_isdir.call_count == 3
        assert mock_sleep.call_count == 2, (
            "must retry (and sleep between attempts) rather than raise "
            "immediately after a single failed recheck"
        )

    def test_no_retry_needed_when_isdir_immediately_true(self):
        """isdir() is True on the very first recheck: succeed without any
        sleep — the common case (a true local race CPython already handles,
        or a fast-converging remote cache) must not be slowed down."""
        from dlio_benchmark.storage.file_storage import _makedirs_race_safe

        with patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=FileExistsError(17, "File exists")), \
             patch("dlio_benchmark.storage.file_storage.os.path.isdir",
                   return_value=True), \
             patch("dlio_benchmark.storage.file_storage.sleep") as mock_sleep:
            _makedirs_race_safe("/shared/ckpt/global_epoch1_step2", exist_ok=True)

        assert mock_sleep.call_count == 0

    # -- must still raise for genuine (non-race) failures ---------------

    def test_reraises_if_never_becomes_a_directory(self):
        """isdir() stays False through every retry (e.g. a stray FILE exists
        at that path, not a directory another rank created) -- exist_ok=True
        must not become a blanket error-suppression switch."""
        from dlio_benchmark.storage.file_storage import (
            _makedirs_race_safe, _MAKEDIRS_RACE_MAX_ATTEMPTS,
        )

        with patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=FileExistsError(17, "File exists")), \
             patch("dlio_benchmark.storage.file_storage.os.path.isdir",
                   return_value=False) as mock_isdir, \
             patch("dlio_benchmark.storage.file_storage.sleep"), \
             pytest.raises(FileExistsError):
            _makedirs_race_safe("/shared/ckpt/stray-file-path", exist_ok=True)

        assert mock_isdir.call_count == _MAKEDIRS_RACE_MAX_ATTEMPTS + 1, (
            "must exhaust all retries (plus the final post-loop check) "
            "before giving up"
        )

    def test_exist_ok_false_reraises_immediately_without_retry(self):
        """exist_ok=False must behave exactly like plain os.makedirs — raise
        immediately, no retry loop, no sleep (the caller explicitly wants an
        error on any existing path)."""
        from dlio_benchmark.storage.file_storage import _makedirs_race_safe

        with patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=FileExistsError(17, "File exists")), \
             patch("dlio_benchmark.storage.file_storage.os.path.isdir") as mock_isdir, \
             patch("dlio_benchmark.storage.file_storage.sleep") as mock_sleep, \
             pytest.raises(FileExistsError):
            _makedirs_race_safe("/shared/ckpt/x", exist_ok=False)

        assert mock_isdir.call_count == 0
        assert mock_sleep.call_count == 0

    def test_other_os_errors_propagate_unchanged(self):
        """Only FileExistsError triggers the retry path — a PermissionError
        (or any other OSError) must propagate immediately, unmodified."""
        from dlio_benchmark.storage.file_storage import _makedirs_race_safe

        with patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=PermissionError(13, "Permission denied")), \
             pytest.raises(PermissionError):
            _makedirs_race_safe("/shared/ckpt/x", exist_ok=True)

    def test_success_path_unaffected(self):
        """The non-racing case (os.makedirs succeeds outright) must be a
        pure passthrough — no isdir recheck, no sleep."""
        from dlio_benchmark.storage.file_storage import _makedirs_race_safe

        with patch("dlio_benchmark.storage.file_storage.os.makedirs") as mock_makedirs, \
             patch("dlio_benchmark.storage.file_storage.os.path.isdir") as mock_isdir, \
             patch("dlio_benchmark.storage.file_storage.sleep") as mock_sleep:
            _makedirs_race_safe("/shared/ckpt/x", exist_ok=True)

        mock_makedirs.assert_called_once_with("/shared/ckpt/x", exist_ok=True)
        assert mock_isdir.call_count == 0
        assert mock_sleep.call_count == 0

    # -- wired into both real call sites ---------------------------------

    def test_create_namespace_uses_race_safe_helper(self):
        """FileStorage.create_namespace must route through the hardened
        helper, not a bare os.makedirs call."""
        from dlio_benchmark.storage.file_storage import FileStorage

        fs = FileStorage.__new__(FileStorage)
        fs.namespace = MagicMock()
        fs.namespace.name = "/shared/ckpt"

        with patch("dlio_benchmark.storage.file_storage._makedirs_race_safe") as mock_fn:
            fs.create_namespace(exist_ok=True)

        mock_fn.assert_called_once_with("/shared/ckpt", True)

    def test_create_node_uses_race_safe_helper(self):
        """FileStorage.create_node must route through the hardened helper,
        not a bare os.makedirs call."""
        from dlio_benchmark.storage.file_storage import FileStorage

        fs = FileStorage.__new__(FileStorage)
        fs.namespace = MagicMock()
        fs.namespace.name = "/shared/ckpt"

        with patch("dlio_benchmark.storage.file_storage._makedirs_race_safe") as mock_fn:
            fs.create_node("global_epoch1_step2", exist_ok=True)

        mock_fn.assert_called_once_with("/shared/ckpt/global_epoch1_step2", True)

    # -- real (non-mocked) concurrency stress test ------------------------

    def test_concurrent_create_namespace_real_threads_no_raise(self):
        """Real FileStorage.create_namespace(), hammered by many threads on
        the SAME new path concurrently. This exercises the true local
        thread/process race CPython's own exist_ok already handles — it
        does not reproduce the networked-filesystem cache-staleness angle
        (not reproducible without a real multi-host NFS/GPFS mount), but it
        does prove the wrapper adds no new local-race regression."""
        from concurrent.futures import ThreadPoolExecutor
        from dlio_benchmark.storage.file_storage import FileStorage

        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "shared", "ckpt", "global_epoch1_step2")
            fs = FileStorage.__new__(FileStorage)
            fs.namespace = MagicMock()
            fs.namespace.name = target

            errors = []

            def _create():
                try:
                    fs.create_namespace(exist_ok=True)
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(lambda _: _create(), range(64)))

            assert not errors, f"concurrent create_namespace raised: {errors}"
            assert os.path.isdir(target)


# ===========================================================================
# storage#699 (sibling) — ObjStoreLibStorage._preflight has the identical
# makedirs race for direct_fs / file-scheme checkpointing
# ===========================================================================
#
# ObjStoreLibStorage._preflight() creates the direct_fs/file-scheme
# namespace directory with the same "bare os.makedirs, swallow OSError,
# recheck isdir()" pattern FileStorage had — vulnerable to the identical
# multi-host/networked-filesystem race. Fixed by routing through the same
# _makedirs_race_safe() helper as FileStorage, rather than duplicating the
# retry logic.


def _make_preflight_instance(uri_scheme, bucket):
    """Construct a minimal ObjStoreLibStorage for exercising _preflight()
    directly, without the full __init__ (credentials/framework machinery
    that direct/file schemes skip anyway)."""
    from dlio_benchmark.storage.obj_store_lib import ObjStoreLibStorage
    from dlio_benchmark.storage.storage_handler import Namespace
    from dlio_benchmark.common.enumerations import NamespaceType

    inst = ObjStoreLibStorage.__new__(ObjStoreLibStorage)
    inst.namespace = Namespace(bucket, NamespaceType.FLAT)
    inst.uri_scheme = uri_scheme
    return inst


class TestStorage699_ObjStoreLibPreflightMakedirsRace:
    """Regression: ObjStoreLibStorage._preflight()'s direct/file-scheme
    directory creation must also survive the multi-host makedirs race
    (storage#699 sibling)."""

    def test_preflight_uses_race_safe_helper(self):
        """_preflight must route directory creation through
        _makedirs_race_safe, not a bare os.makedirs call."""
        from dlio_benchmark.storage.file_storage import (
            _makedirs_race_safe as real_makedirs_race_safe,
        )

        with tempfile.TemporaryDirectory() as d:
            # _preflight only auto-creates when the bucket's IMMEDIATE parent
            # already exists (it does not recursively create ancestors) —
            # so pre-create that parent, not just some higher-up directory.
            parent_of_bucket = os.path.join(d, "parent", "ckpt")
            os.makedirs(parent_of_bucket)
            bucket = os.path.join(parent_of_bucket, "llama3-70b")
            inst = _make_preflight_instance("file", bucket)

            with patch("dlio_benchmark.storage.obj_store_lib._makedirs_race_safe",
                       wraps=real_makedirs_race_safe) as mock_fn:
                inst._preflight()

            mock_fn.assert_called_once_with(bucket, exist_ok=True)
            assert os.path.isdir(bucket)

    def test_preflight_survives_stale_cache_then_converges(self):
        """Same race as FileStorage.create_node(): os.makedirs raises
        FileExistsError, isdir() is False on the first retry-loop recheck
        (stale client cache) then True — _preflight must not raise."""
        bucket = "/shared/ckpt/llama3-70b"

        # os.path.isdir() is called for several distinct purposes inside
        # _preflight + _makedirs_race_safe (does bucket exist? does its
        # parent exist? the retry-loop recheck; the final recheck). Only the
        # retry-loop recheck should see the "stale" False — everything else
        # in this scenario is genuinely True/exists.
        call_count = {"n": 0}

        def isdir_side_effect(path):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                return False   # bucket doesn't exist yet
            if n == 2:
                return True    # bucket's parent directory exists
            if n == 3:
                return False   # first retry-loop recheck: stale client cache
            return True        # converged: every later recheck sees it

        with patch("dlio_benchmark.storage.obj_store_lib.os.path.isdir",
                   side_effect=isdir_side_effect), \
             patch("dlio_benchmark.storage.obj_store_lib.os.path.dirname",
                   return_value="/shared/ckpt"), \
             patch("dlio_benchmark.storage.obj_store_lib.os.access",
                   return_value=True), \
             patch("dlio_benchmark.storage.file_storage.os.makedirs",
                   side_effect=FileExistsError(17, "File exists")), \
             patch("dlio_benchmark.storage.file_storage.sleep") as mock_sleep:
            inst = _make_preflight_instance("file", bucket)
            inst._preflight()  # must not raise

        assert mock_sleep.call_count == 1, (
            "must have gone through the retry path (one stale recheck, one "
            "sleep, then converged) rather than raising on the first recheck"
        )

    def test_direct_scheme_also_covered(self):
        """The direct:// (O_DIRECT) scheme shares the same code path as
        file:// — must be covered too, not just file://."""
        with tempfile.TemporaryDirectory() as d:
            parent_of_bucket = os.path.join(d, "parent", "ckpt")
            os.makedirs(parent_of_bucket)
            bucket = os.path.join(parent_of_bucket, "llama3-70b")
            inst = _make_preflight_instance("direct", bucket)
            inst._preflight()  # must not raise
            assert os.path.isdir(bucket)


# ===========================================================================
# storage#699 (sibling) — SimpleStreamingCheckpointing.save() has the
# identical makedirs race, in the fallback checkpoint backend used when
# mlpstorage is not installed
# ===========================================================================


class TestStorage699_SimpleStreamingCheckpointingMakedirsRace:
    """Regression: the mlpstorage-unavailable fallback checkpoint writer must
    also survive the multi-host makedirs race (storage#699 sibling) — a run
    without mlpstorage installed would otherwise hit the exact reported bug
    unfixed."""

    def test_save_uses_race_safe_helper(self):
        from dlio_benchmark.checkpointing.simple_streaming_checkpointing import (
            SimpleStreamingCheckpointing,
        )
        from dlio_benchmark.checkpointing import simple_streaming_checkpointing as mod

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt", "global_epoch1_step2", "rank0.bin")
            ckpt = SimpleStreamingCheckpointing(chunk_size=1024)

            with patch.object(mod, "_makedirs_race_safe",
                              wraps=mod._makedirs_race_safe) as mock_fn:
                ckpt.save(path, total_size_bytes=2048)

            mock_fn.assert_called_once_with(os.path.dirname(path), exist_ok=True)
            assert os.path.isfile(path)
            assert os.path.getsize(path) == 2048

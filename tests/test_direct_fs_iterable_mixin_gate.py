"""
Regression tests for the O_DIRECT prefetch-path gate in
``_LocalFSIterableMixin._localfs_init``.

Background — mlcommons/storage#567
==================================
Before this fix, the mixin only selected the s3dlio O_DIRECT path when
``storage_options.storage_library == "direct"``. But ``storage_type=direct_fs``
(reached from mlpstorage's ``--o-direct``) is *required* by
``utils/config.py`` to set ``storage_library == "s3dlio"``. With the old
``lib == "direct"`` gate that meant ``_use_direct`` was always ``False`` for
``direct_fs`` runs — the mixin then handed ``direct://…`` URIs to plain
``open()`` and crashed with ``FileNotFoundError`` on the warmup batch for
every local NPZ / NPY / JPEG workload (UNet3D, ResNet, …).

The fix accepts EITHER signal:
  - ``uri_scheme == "direct"``  (canonical — matches direct_fs validation)
  - ``storage_library == "direct"``  (legacy single-knob form)

These tests lock both signals so the gate can't silently drift back to the
single-knob check that broke direct_fs.
"""
import pytest
from unittest.mock import MagicMock

from dlio_benchmark.reader._local_fs_iterable_mixin import _LocalFSIterableMixin


def _make_instance(storage_options):
    """Build a bare mixin instance whose ``_args.storage_options`` returns
    the given dict. We bypass ``__init__`` on purpose — ``_localfs_init`` is
    the only method under test, and the full reader stack pulls in mpi4py
    + torch + hydra which is overkill for a sentinel-comparison unit test.
    """
    instance = _LocalFSIterableMixin.__new__(_LocalFSIterableMixin)
    instance._args = MagicMock()
    instance._args.storage_options = storage_options
    return instance


class TestUseDirectGateForDirectFsConfig:
    """storage#567: configs emitted by ``storage_type=direct_fs`` MUST take
    the O_DIRECT path."""

    def test_direct_fs_shape_selects_direct(self):
        """The exact config shape mlpstorage's ``--o-direct`` produces:
        storage_library=s3dlio + uri_scheme=direct. Pre-fix this was the
        broken case — ``_use_direct`` was False and the mixin handed
        ``direct://…`` to ``open()``."""
        inst = _make_instance({"storage_library": "s3dlio", "uri_scheme": "direct"})
        inst._localfs_init()
        assert inst._use_direct is True, (
            "storage#567 regression: storage_library=s3dlio + uri_scheme=direct "
            "MUST select the O_DIRECT path. _use_direct=False means the "
            "buffered open() path will be invoked on direct:// URIs."
        )

    def test_uri_scheme_direct_alone_selects_direct(self):
        """uri_scheme is sufficient on its own — storage_library may be
        absent or set to any non-'direct' value."""
        inst = _make_instance({"uri_scheme": "direct"})
        inst._localfs_init()
        assert inst._use_direct is True


class TestUseDirectGateLegacyForm:
    """Backward compatibility: the legacy single-knob form must keep
    working."""

    def test_storage_library_direct_alone_selects_direct(self):
        inst = _make_instance({"storage_library": "direct"})
        inst._localfs_init()
        assert inst._use_direct is True

    def test_both_knobs_set_to_direct_selects_direct(self):
        inst = _make_instance({"storage_library": "direct", "uri_scheme": "direct"})
        inst._localfs_init()
        assert inst._use_direct is True


class TestUseDirectGateBufferedFallback:
    """The buffered path must remain the default for non-direct configs."""

    def test_empty_options_selects_buffered(self):
        inst = _make_instance({})
        inst._localfs_init()
        assert inst._use_direct is False

    def test_none_options_selects_buffered(self):
        """getattr returning None must not blow up; treat as buffered."""
        instance = _LocalFSIterableMixin.__new__(_LocalFSIterableMixin)
        instance._args = MagicMock()
        instance._args.storage_options = None
        instance._localfs_init()
        assert instance._use_direct is False

    def test_posix_storage_library_selects_buffered(self):
        inst = _make_instance({"storage_library": "posix"})
        inst._localfs_init()
        assert inst._use_direct is False

    def test_s3dlio_with_s3_uri_scheme_selects_buffered(self):
        """storage_library=s3dlio is also used for real S3 (uri_scheme=s3).
        That must NOT trigger the local O_DIRECT path."""
        inst = _make_instance({"storage_library": "s3dlio", "uri_scheme": "s3"})
        inst._localfs_init()
        assert inst._use_direct is False, (
            "storage_library=s3dlio alone must not imply local O_DIRECT — "
            "the same library serves remote S3 with uri_scheme=s3."
        )

    def test_s3dlio_with_file_uri_scheme_selects_buffered(self):
        """file:// is a separate s3dlio mode (buffered local FS via s3dlio);
        not the O_DIRECT path."""
        inst = _make_instance({"storage_library": "s3dlio", "uri_scheme": "file"})
        inst._localfs_init()
        assert inst._use_direct is False


class TestUseDirectInitsCounters:
    """The init must always seed the bookkeeping attributes the rest of
    the mixin reads, regardless of which path is selected."""

    @pytest.mark.parametrize("opts", [
        {"storage_library": "s3dlio", "uri_scheme": "direct"},
        {"storage_library": "direct"},
        {},
        {"storage_library": "posix"},
    ])
    def test_counters_seeded(self, opts):
        inst = _make_instance(opts)
        inst._localfs_init()
        assert inst._local_cache == {}
        assert inst._total_bytes_read == 0
        assert inst._total_objects_read == 0

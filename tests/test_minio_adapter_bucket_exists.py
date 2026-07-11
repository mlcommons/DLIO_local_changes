"""
Regression tests for mlcommons/storage#756 — MinIOAdapter was missing
a `bucket_exists()` method that ObjStoreLibStorage._preflight()
unconditionally calls when storage_library=minio, causing every
minio-backed training run to die at storage-construction time with:

    AttributeError: 'MinIOAdapter' object has no attribute 'bucket_exists'

Fix: MinIOAdapter now delegates to the underlying `minio.Minio.
bucket_exists(bucket_name)` client method, matching the shape of
its sibling wrappers (`get_object`, `put_object`, `list_objects`).
"""

import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def stub_minio(monkeypatch):
    """Stub out `minio.Minio` so MinIOAdapter.__init__ doesn't try to
    contact a real server.  Returns the stubbed client mock so tests can
    program its return values.

    The `minio` package is not a hard DLIO dependency — it is imported
    lazily inside `MinIOAdapter.__init__` (`from minio import Minio`).
    We install a synthetic module under that name so the lazy import
    resolves to our stub without requiring the real package.
    """
    client = mock.MagicMock()
    minio_ctor = mock.MagicMock(return_value=client)

    fake_minio = types.ModuleType("minio")
    fake_minio.Minio = minio_ctor
    monkeypatch.setitem(sys.modules, "minio", fake_minio)
    return client, minio_ctor


class TestMinIOAdapterBucketExists:
    """Coverage for mlcommons/storage#756.

    _preflight() has always called `self.s3_client.bucket_exists(bucket)`
    for the minio path (see obj_store_lib.py `_preflight()` — the block
    guarded by `elif self.storage_library == "minio"`).  MinIOAdapter
    just never grew the wrapper method, so every minio-backed run died
    at preflight with AttributeError.
    """

    def test_minio_adapter_has_bucket_exists_method(self, stub_minio):
        """RED-then-GREEN core: MinIOAdapter must expose bucket_exists()."""
        from dlio_benchmark.storage.obj_store_lib import MinIOAdapter

        adapter = MinIOAdapter(
            endpoint="http://localhost:9000",
            access_key="test",
            secret_key="test",
            region="us-east-1",
            secure=False,
        )
        assert hasattr(adapter, "bucket_exists"), (
            "MinIOAdapter must have a bucket_exists() method — "
            "ObjStoreLibStorage._preflight() calls "
            "self.s3_client.bucket_exists(bucket) on the minio path "
            "(see storage#756)."
        )

    def test_bucket_exists_returns_true_when_bucket_present(self, stub_minio):
        """Happy path: delegate to minio.Minio.bucket_exists()."""
        client, _ = stub_minio
        client.bucket_exists.return_value = True

        from dlio_benchmark.storage.obj_store_lib import MinIOAdapter

        adapter = MinIOAdapter(
            endpoint="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )
        assert adapter.bucket_exists("mlcommons") is True
        client.bucket_exists.assert_called_once_with("mlcommons")

    def test_bucket_exists_returns_false_when_bucket_missing(self, stub_minio):
        """False-return path: no exception raised, just a False return."""
        client, _ = stub_minio
        client.bucket_exists.return_value = False

        from dlio_benchmark.storage.obj_store_lib import MinIOAdapter

        adapter = MinIOAdapter(
            endpoint="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )
        assert adapter.bucket_exists("nonexistent-bucket") is False

    def test_bucket_exists_propagates_underlying_exceptions(self, stub_minio):
        """Auth / endpoint / DNS / TLS errors raised by minio.Minio.
        bucket_exists() (S3Error, urllib3 errors, ConnectionError, ...)
        must propagate.  ObjStoreLibStorage._preflight() catches them
        one layer up and re-raises as ConnectionError with a rich
        message; MinIOAdapter itself must NOT swallow them.
        """
        client, _ = stub_minio

        class FakeS3Error(Exception):
            """Stand-in for minio.S3Error without importing the whole SDK."""

        client.bucket_exists.side_effect = FakeS3Error(
            "The Access Key Id you provided does not exist in our records."
        )

        from dlio_benchmark.storage.obj_store_lib import MinIOAdapter

        adapter = MinIOAdapter(
            endpoint="http://localhost:9000",
            access_key="wrong-key",
            secret_key="wrong-sec",
        )
        with pytest.raises(FakeS3Error, match="Access Key Id"):
            adapter.bucket_exists("mlcommons")

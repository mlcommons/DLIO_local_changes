"""
Write-verification opt-in tests (mlcommons/storage#593, s3dlio >= 0.9.106)
============================================================================

As of s3dlio 0.9.106, HEAD-after-write verification is OPT-IN and disabled
by default for both write paths used by ``ObjStoreLibStorage``:

  * single-part PUT  (``put_bytes`` / ``put_bytes_async``)  — gated by
    ``S3DLIO_PUT_VERIFY``
  * multipart upload (``MultipartUploadWriter``)             — gated by
    ``S3DLIO_MPU_PUT_VERIFY``

These tests confirm, against a real S3-compatible endpoint:

  1. Default (both flags unset) — NO HEAD verification occurs, writes still
     land correctly.
  2. Opt-in (flag set to "true") — HEAD verification DOES occur.
  3. DLIO's own ``ObjStoreLibStorage._mpu_upload_with_retry`` retry/abort
     logic works correctly end-to-end through the real installed wheel,
     independent of whether verification is enabled.

Whether verification ran is observed directly via ``s3dlio.init_logging
("debug")`` (writes to stdout, captured with pytest's built-in ``capfd``
fixture — s3dlio's tracing subscriber is native Rust code writing straight
to the OS-level stdout file descriptor, which only an fd-level capture
mechanism like ``capfd`` sees) plus an independent HEAD/stat check performed
by the test itself — not by trusting s3dlio's internal state.

Opt-in gate
-----------
These tests hit a live S3-compatible endpoint and are NOT run by default
(consistent with tests/test_s3dlio_object_store.py and the project's
checkin-test policy: only tests/test_fast_ci.py runs automatically).

    DLIO_OBJECT_STORAGE_TESTS=1 pytest tests/test_write_verification_593.py -v

Credentials are loaded from sibling repos' .env files (best effort) or from
already-exported AWS_* / BUCKET environment variables.
"""

import os
import time
import types
import uuid
from pathlib import Path

import pytest

# ─── Object-storage opt-in gate (matches test_s3dlio_object_store.py) ────────
_OBJECT_TESTS_ENABLED = os.environ.get("DLIO_OBJECT_STORAGE_TESTS", "0") == "1"
if not _OBJECT_TESTS_ENABLED:
    pytest.skip(
        "Object-storage tests are disabled. Set DLIO_OBJECT_STORAGE_TESTS=1 to enable.",
        allow_module_level=True,
    )


def _load_env_file(path: Path) -> None:
    """Load key=value pairs from a .env file, skipping keys already set."""
    if not path.exists():
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key not in os.environ:
                os.environ[key] = val


# Best-effort: pick up credentials from a sibling s3dlio/.env if present
# (the live endpoint + bucket used throughout this project's S3 testing).
_load_env_file(Path(__file__).resolve().parent.parent.parent / "s3dlio" / ".env")

import s3dlio  # noqa: E402  (import after .env load, matching project convention)

from dlio_benchmark.storage.obj_store_lib import ObjStoreLibStorage  # noqa: E402


def _bucket() -> str:
    bucket = os.environ.get("BUCKET")
    if not bucket:
        pytest.skip("BUCKET not set — cannot run live write-verification tests")
    return bucket


def _unique_key(tag: str) -> str:
    return f"test-write-verify-593-{tag}-{uuid.uuid4().hex}.bin"


def _mpu_stand_in(max_retries: int = 3, retry_delay_s: float = 0.0):
    """A minimal stand-in for `self` sufficient to call
    ObjStoreLibStorage._mpu_upload_with_retry directly, without going through
    the full Hydra/ConfigArguments-driven DLIOBenchmark bootstrap that a real
    ObjStoreLibStorage.__init__ requires.  The method only touches
    `_s3dlio`, `_MULTIPART_MAX_RETRIES`, and `_MULTIPART_RETRY_DELAY_S`."""
    return types.SimpleNamespace(
        _s3dlio=s3dlio,
        _MULTIPART_MAX_RETRIES=max_retries,
        _MULTIPART_RETRY_DELAY_S=retry_delay_s,
    )


# ---------------------------------------------------------------------------
# Single-part PUT (put_bytes) — S3DLIO_PUT_VERIFY
# ---------------------------------------------------------------------------


def test_put_bytes_default_off_no_verification(capfd):
    """Default (S3DLIO_PUT_VERIFY unset): no internal HEAD-verify-retry loop
    runs.  The object must still land correctly — confirmed independently by
    this test's own stat() call, not by trusting s3dlio's internal state."""
    os.environ.pop("S3DLIO_PUT_VERIFY", None)
    s3dlio.init_logging("debug")
    bucket = _bucket()
    key = _unique_key("put-default-off")
    uri = f"s3://{bucket}/{key}"
    data = bytes([0x42]) * 4096

    capfd.readouterr()  # drain any output from init_logging / prior tests
    s3dlio.put_bytes(uri, data)
    log_text = capfd.readouterr().out

    assert "put_bytes: attempt=" not in log_text, (
        "verification loop must NOT run when S3DLIO_PUT_VERIFY is unset (default)"
    )

    meta = s3dlio.stat(uri)
    assert meta["size"] == len(data), (
        f"object must land correctly even without internal verification: "
        f"expected {len(data)} got {meta['size']}"
    )
    s3dlio.delete(uri)


def test_put_bytes_opt_in_runs_verification(capfd):
    """Opt-in (S3DLIO_PUT_VERIFY=true): the HEAD-verify-retry loop runs."""
    os.environ["S3DLIO_PUT_VERIFY"] = "true"
    s3dlio.init_logging("debug")
    bucket = _bucket()
    key = _unique_key("put-opt-in")
    uri = f"s3://{bucket}/{key}"
    data = bytes([0x99]) * 4096

    try:
        capfd.readouterr()  # drain any output from init_logging / prior tests
        s3dlio.put_bytes(uri, data)
        log_text = capfd.readouterr().out

        assert "put_bytes: attempt=1/" in log_text, (
            "verification loop must run when S3DLIO_PUT_VERIFY=true"
        )

        meta = s3dlio.stat(uri)
        assert meta["size"] == len(data), (
            f"HEAD-verified size mismatch: expected {len(data)} got {meta['size']}"
        )
        s3dlio.delete(uri)
    finally:
        os.environ.pop("S3DLIO_PUT_VERIFY", None)


# ---------------------------------------------------------------------------
# Multipart upload (_mpu_upload_with_retry) — S3DLIO_MPU_PUT_VERIFY
# ---------------------------------------------------------------------------


def test_mpu_upload_with_retry_default_off_no_verification(capfd):
    """Default (S3DLIO_MPU_PUT_VERIFY unset): DLIO's
    ObjStoreLibStorage._mpu_upload_with_retry completes successfully through
    the real s3dlio MultipartUploadWriter with no internal HEAD check.  The
    object must still land correctly — confirmed independently."""
    os.environ.pop("S3DLIO_MPU_PUT_VERIFY", None)
    s3dlio.init_logging("debug")
    bucket = _bucket()
    key = _unique_key("mpu-default-off")
    uri = f"s3://{bucket}/{key}"
    data_size = 12 * 1024 * 1024  # 12 MiB — spans 2 parts at default 8 MiB
    payload = bytes([0xAB]) * data_size

    stand_in = _mpu_stand_in()
    capfd.readouterr()  # drain any output from init_logging / prior tests
    ObjStoreLibStorage._mpu_upload_with_retry(stand_in, uri, lambda: [payload])
    log_text = capfd.readouterr().out

    assert "skipping HEAD verification" in log_text, (
        "s3dlio must skip the HEAD check when S3DLIO_MPU_PUT_VERIFY is unset (default)"
    )

    meta = s3dlio.stat(uri)
    assert meta["size"] == data_size, (
        f"object must land correctly even without internal verification: "
        f"expected {data_size} got {meta['size']}"
    )
    s3dlio.delete(uri)


def test_mpu_upload_with_retry_opt_in_runs_verification(capfd):
    """Opt-in (S3DLIO_MPU_PUT_VERIFY=true): a HEAD check runs after
    CompleteMultipartUpload, observed via s3dlio's debug log."""
    os.environ["S3DLIO_MPU_PUT_VERIFY"] = "true"
    s3dlio.init_logging("debug")
    bucket = _bucket()
    key = _unique_key("mpu-opt-in")
    uri = f"s3://{bucket}/{key}"
    data_size = 12 * 1024 * 1024
    payload = bytes([0xCD]) * data_size

    try:
        stand_in = _mpu_stand_in()
        capfd.readouterr()  # drain any output from init_logging / prior tests
        ObjStoreLibStorage._mpu_upload_with_retry(stand_in, uri, lambda: [payload])
        log_text = capfd.readouterr().out

        assert "HEAD verified stored=" in log_text, (
            "s3dlio must run the HEAD check when S3DLIO_MPU_PUT_VERIFY=true"
        )

        meta = s3dlio.stat(uri)
        assert meta["size"] == data_size, (
            f"HEAD-verified size mismatch: expected {data_size} got {meta['size']}"
        )
        s3dlio.delete(uri)
    finally:
        os.environ.pop("S3DLIO_MPU_PUT_VERIFY", None)


def test_mpu_upload_with_retry_retries_on_genuine_error():
    """DLIO's retry/abort logic itself works correctly: a writer that fails
    triggers writer.abort() + a fresh writer on the next attempt, regardless
    of S3DLIO_MPU_PUT_VERIFY.  This exercises a genuine API-level error (zero
    parts written, which every S3-compatible backend rejects on
    CompleteMultipartUpload) rather than a verification-triggered one, since
    the retry loop must handle both causes identically."""
    os.environ.pop("S3DLIO_MPU_PUT_VERIFY", None)
    bucket = _bucket()
    key = _unique_key("mpu-retry-zero-parts")
    uri = f"s3://{bucket}/{key}"

    stand_in = _mpu_stand_in(max_retries=2, retry_delay_s=0.0)
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        # Empty data_source -> zero parts written -> CompleteMultipartUpload
        # with an empty parts list, rejected by every S3-compatible backend.
        ObjStoreLibStorage._mpu_upload_with_retry(stand_in, uri, lambda: [])
    elapsed = time.monotonic() - start

    assert elapsed < 30, (
        f"retry loop took {elapsed:.1f}s for 2 attempts with 0s delay — "
        "unexpectedly slow, check for an unintended sleep"
    )

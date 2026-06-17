"""
Tests for SyntheticFileList — lazy file-list container that avoids
materialising millions of Python str objects per rank, fixing OOM on
memory-constrained nodes.

Verifies:
  * SyntheticFileList path layout matches data_generator.py exactly
    (both with and without subfolders).
  * SyntheticFileList is stateless: shuffle() is a no-op (shuffling
    is delegated to VirtualIndexMap._sample_list) and slicing returns
    a working view.
  * validate_on_disk() catches missing files and reports useful errors.
  * VirtualIndexMap still works when fed a SyntheticFileList.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dlio_benchmark.utils.file_index import (
    SyntheticFileList,
    shuffle_file_list,
)


# ---------------------------------------------------------------------------
# Reference path formatter that mirrors data_generator.py exactly
# ---------------------------------------------------------------------------

def _datagen_path(data_dir, dataset_type, i, num_files, num_subfolders,
                  file_prefix="img", file_format="npz"):
    """Reproduce dlio_benchmark.data_generator.data_generator.generate()."""
    nd_f = len(str(num_files))
    if num_subfolders > 1:
        nd_sf = len(str(num_subfolders))
        return "{}/{}/{}/{}_{}_of_{}.{}".format(
            data_dir, dataset_type,
            str(i % num_subfolders).zfill(nd_sf),
            file_prefix,
            str(i).zfill(nd_f),
            num_files, file_format,
        )
    return "{}/{}/{}_{}_of_{}.{}".format(
        data_dir, dataset_type,
        file_prefix,
        str(i).zfill(nd_f),
        num_files, file_format,
    )


# ---------------------------------------------------------------------------
# SyntheticFileList correctness
# ---------------------------------------------------------------------------

class TestSyntheticFileListLayout:
    """Path layout must exactly match data_generator.py output."""

    @pytest.mark.parametrize("n,subs,fmt,prefix", [
        (100, 0, "npz", "img"),
        (100, 1, "npz", "img"),
        (1_000, 10, "jpeg", "img"),
        (1_000, 13, "npz", "img"),
        (7_646_857, 764, "jpeg", "img"),
        (133_301, 13, "npz", "img"),
        (33_691, 0, "h5", "img"),
        (50_000, 100, "tfrecord", "part"),
    ])
    def test_path_format_matches_datagen(self, n, subs, fmt, prefix):
        sfl = SyntheticFileList("/data/root", "train", n, subs,
                                file_prefix=prefix, file_format=fmt)
        # Identity check: first, last, a stride sample, and a middle
        for i in [0, 1, n // 2, n - 1]:
            assert sfl[i] == _datagen_path(
                "/data/root", "train", i, n, subs,
                file_prefix=prefix, file_format=fmt
            ), f"mismatch at i={i} for n={n}, subs={subs}"

    def test_length(self):
        sfl = SyntheticFileList("/d", "train", 1234, 7)
        assert len(sfl) == 1234

    def test_iter(self):
        sfl = SyntheticFileList("/d", "train", 50, 5, file_format="npz")
        paths = list(sfl)
        assert len(paths) == 50
        # First yielded path is the formatted version of perm[0] (=0 before shuffle)
        assert paths[0] == _datagen_path("/d", "train", 0, 50, 5)
        assert paths[-1] == _datagen_path("/d", "train", 49, 50, 5)

    def test_no_subfolders_when_subs_le_1(self):
        """When num_subfolders is 0 or 1, no '/N/' segment appears."""
        for subs in (0, 1):
            sfl = SyntheticFileList("/d", "train", 10, subs)
            p = sfl[0]
            assert "/train/" in p
            # No second-level integer directory between 'train' and the filename
            parts = p.split("/")
            assert parts[-2] == "train", \
                f"unexpected subfolder for subs={subs}: {p}"


# ---------------------------------------------------------------------------
# SyntheticFileList shuffle / slice semantics
# ---------------------------------------------------------------------------

class TestSyntheticFileListMutation:

    def test_shuffle_is_a_noop(self):
        """Stateless file list: shuffle MUST NOT reorder paths.

        Randomisation of access order is delegated to
        VirtualIndexMap._sample_list so that this object's memory cost
        does not scale with num_files_total.  shuffle() exists only
        for API compatibility with the legacy list[str] path; it
        accepts the seed and does nothing.
        """
        n = 1000
        sfl = SyntheticFileList("/d", "train", n, 10)
        before = [sfl[i] for i in range(n)]
        sfl.shuffle(seed=42)
        after = [sfl[i] for i in range(n)]
        assert before == after, (
            "stateless SyntheticFileList.shuffle must be a no-op; "
            "shuffling is delegated to VirtualIndexMap._sample_list"
        )

    def test_shuffle_is_reproducible(self):
        """Two instances must always produce identical sequences."""
        sfl1 = SyntheticFileList("/d", "train", 500, 5)
        sfl2 = SyntheticFileList("/d", "train", 500, 5)
        sfl1.shuffle(seed=7); sfl2.shuffle(seed=7)
        assert [sfl1[i] for i in range(500)] == [sfl2[i] for i in range(500)]

    def test_slice_returns_view(self):
        sfl = SyntheticFileList("/d", "train", 100, 5)
        sub = sfl[10:20]
        assert isinstance(sub, SyntheticFileList)
        assert len(sub) == 10
        assert sub[0] == sfl[10]
        assert sub[-1] == sfl[19]

    def test_slice_after_shuffle_is_unchanged(self):
        """shuffle() is a no-op, so slice contents are deterministic."""
        sfl = SyntheticFileList("/d", "train", 100, 5)
        sfl.shuffle(seed=1)
        sub = sfl[:50]
        assert len(sub) == 50
        for i in range(50):
            assert sub[i] == sfl[i]

    def test_slice_with_stride(self):
        sfl = SyntheticFileList("/d", "train", 100, 5)
        sub = sfl[10:50:3]
        expected = [sfl[i] for i in range(10, 50, 3)]
        assert len(sub) == len(expected)
        assert [sub[i] for i in range(len(sub))] == expected

    def test_slice_of_slice(self):
        """Composing slices must compose start/stride correctly."""
        sfl = SyntheticFileList("/d", "train", 100, 5)
        sub1 = sfl[10:90]            # start=10, stride=1, len=80
        sub2 = sub1[5:25]            # start=15, stride=1, len=20
        for i in range(20):
            assert sub2[i] == sfl[15 + i]

    def test_shuffle_via_helper_is_a_noop(self):
        """shuffle_file_list helper dispatches to .shuffle() which is no-op."""
        sfl = SyntheticFileList("/d", "train", 200, 4)
        before = [sfl[i] for i in range(200)]
        shuffle_file_list(sfl, seed=99)
        after = [sfl[i] for i in range(200)]
        assert before == after, (
            "shuffle_file_list on stateless SyntheticFileList must be a no-op"
        )

    def test_shuffle_via_helper_legacy_list(self):
        """shuffle_file_list helper must still work for plain lists."""
        lst = [f"/d/{i}" for i in range(200)]
        original = list(lst)
        shuffle_file_list(lst, seed=99)
        assert sorted(lst) == sorted(original)
        assert lst != original  # very high probability at n=200


# ---------------------------------------------------------------------------
# Memory usage sanity check
# ---------------------------------------------------------------------------

class TestSyntheticFileListMemory:

    def test_no_perm_array_allocated(self):
        """Stateless: no O(N) allocation regardless of dataset size.

        At 100M files the legacy permutation array alone would be 800 MB.
        With Option A the instance has only scalar slots; size << 1 KB.
        """
        import sys as _sys
        # No N-scaling allocations
        sfl = SyntheticFileList("/data/root", "train", 100_000_000, 1000,
                                file_prefix="img", file_format="jpeg")
        assert not hasattr(sfl, "_permutation"), (
            "Option A: _permutation must not exist (stateless file list)"
        )
        # __slots__ keeps the instance footprint tiny.  The instance dict
        # is absent; only the scalar slot pointers count.
        assert _sys.getsizeof(sfl) < 1024
        # Lookups still work at this scale
        path = sfl[50_000_000]
        assert "_of_100000000." in path

    def test_memory_does_not_grow_with_n(self):
        """Sanity: a 1k-file and 100M-file list have the same sizeof."""
        import sys as _sys
        small = SyntheticFileList("/d", "train", 1_000, 10)
        huge = SyntheticFileList("/d", "train", 100_000_000, 1000)
        assert _sys.getsizeof(small) == _sys.getsizeof(huge)


# ---------------------------------------------------------------------------
# validate_on_disk
# ---------------------------------------------------------------------------

class TestSyntheticFileListValidate:

    def test_validate_passes_when_files_exist(self, tmp_path):
        # Create a tiny synthetic dataset on disk
        n = 32
        subs = 4
        root = tmp_path / "train"
        for sub in range(subs):
            (root / f"{sub}").mkdir(parents=True)
        for i in range(n):
            p = _datagen_path(str(tmp_path), "train", i, n, subs,
                              file_prefix="img", file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00" * 64)

        sfl = SyntheticFileList(str(tmp_path), "train", n, subs,
                                file_prefix="img", file_format="bin")
        probed = sfl.validate_on_disk(expected_record_bytes=32,
                                       require_size_match=True)
        assert len(probed) >= 2  # at least first + last
        assert all(sz >= 32 for _, _, sz in probed)

    def test_validate_raises_when_first_missing(self, tmp_path):
        sfl = SyntheticFileList(str(tmp_path), "train", 10, 0,
                                file_prefix="img", file_format="bin")
        with pytest.raises(FileNotFoundError) as exc:
            sfl.validate_on_disk()
        assert "probe failed at index 0" in str(exc.value)

    def test_validate_raises_when_last_missing(self, tmp_path):
        n = 10
        # Create only first file
        (tmp_path / "train").mkdir()
        first = _datagen_path(str(tmp_path), "train", 0, n, 0,
                              file_prefix="img", file_format="bin")
        with open(first, "wb") as f:
            f.write(b"x")
        sfl = SyntheticFileList(str(tmp_path), "train", n, 0,
                                file_prefix="img", file_format="bin")
        with pytest.raises(FileNotFoundError) as exc:
            sfl.validate_on_disk(num_random_probes=0)
        assert f"probe failed at index {n-1}" in str(exc.value)

    def test_validate_size_mismatch(self, tmp_path):
        (tmp_path / "train").mkdir()
        for i in range(2):
            p = _datagen_path(str(tmp_path), "train", i, 2, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"x")  # 1 byte
        sfl = SyntheticFileList(str(tmp_path), "train", 2, 0, file_format="bin")
        with pytest.raises(ValueError, match="found 1 B"):
            sfl.validate_on_disk(expected_record_bytes=100,
                                  require_size_match=True, num_random_probes=0)


# ---------------------------------------------------------------------------
# peek_total_from_disk + subset-by-truncation semantics
# ---------------------------------------------------------------------------

class TestSyntheticFileListPeekAndSubset:
    """Reusing a larger generated dataset: discover N from disk, slice
    down to the requested num_files_train."""

    def _populate(self, tmp_path, n, subs, file_format="bin", file_prefix="img"):
        if subs > 1:
            for sub in range(subs):
                width = len(str(subs))
                (tmp_path / "train" / str(sub).zfill(width)).mkdir(parents=True)
        else:
            (tmp_path / "train").mkdir()
        for i in range(n):
            p = _datagen_path(str(tmp_path), "train", i, n, subs,
                              file_prefix=file_prefix, file_format=file_format)
            with open(p, "wb") as f:
                f.write(b"\x00" * 8)

    def test_peek_total_with_subfolders(self, tmp_path):
        self._populate(tmp_path, n=100, subs=10, file_format="npz")
        total = SyntheticFileList.peek_total_from_disk(
            data_folder=str(tmp_path), dataset_type="train",
            num_subfolders=10, file_prefix="img", file_format="npz")
        assert total == 100

    def test_peek_total_without_subfolders(self, tmp_path):
        self._populate(tmp_path, n=50, subs=0, file_format="bin")
        total = SyntheticFileList.peek_total_from_disk(
            data_folder=str(tmp_path), dataset_type="train",
            num_subfolders=0, file_prefix="img", file_format="bin")
        assert total == 50

    def test_peek_total_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            SyntheticFileList.peek_total_from_disk(
                data_folder=str(tmp_path), dataset_type="train",
                num_subfolders=0, file_prefix="img", file_format="bin")

    def test_peek_total_empty_dir_raises(self, tmp_path):
        (tmp_path / "train").mkdir()
        with pytest.raises(FileNotFoundError, match="no file matching"):
            SyntheticFileList.peek_total_from_disk(
                data_folder=str(tmp_path), dataset_type="train",
                num_subfolders=0, file_prefix="img", file_format="bin")

    def test_subset_preserves_suffix_and_files_exist(self, tmp_path):
        """The crucial property: even after slicing to a smaller view,
        the _of_N suffix and the resolved paths still match files on
        disk that were generated for the original (larger) N."""
        on_disk_n = 100
        self._populate(tmp_path, n=on_disk_n, subs=5, file_format="bin")
        # User wants only 25 files for training
        requested = 25
        full = SyntheticFileList(str(tmp_path), "train",
                                 num_files=on_disk_n, num_subfolders=5,
                                 file_prefix="img", file_format="bin")
        subset = full[:requested]
        assert len(subset) == requested
        # Suffix unchanged (= on-disk N)
        assert f"_of_{on_disk_n}.bin" in subset[0]
        assert f"_of_{on_disk_n}.bin" in subset[-1]
        # Every path in the subset must exist on disk
        for path in subset:
            assert os.path.exists(path), f"missing: {path}"
        # validate_on_disk on the FULL list still passes
        full.validate_on_disk(num_random_probes=4)

    def test_subset_indices_match_first_n_logical(self, tmp_path):
        """Slicing [:n] returns first-n logical files (0..n-1)."""
        on_disk_n = 64
        self._populate(tmp_path, n=on_disk_n, subs=4, file_format="bin")
        full = SyntheticFileList(str(tmp_path), "train",
                                 num_files=on_disk_n, num_subfolders=4,
                                 file_prefix="img", file_format="bin")
        subset = full[:10]
        for i in range(10):
            expected = _datagen_path(str(tmp_path), "train", i,
                                     on_disk_n, 4, file_format="bin")
            assert subset[i] == expected

    def test_peek_mixed_series_raises(self, tmp_path):
        """Two datagen runs into the same dir must be detected, not guessed."""
        (tmp_path / "train").mkdir()
        # Series A: N=100
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 100, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        # Series B: N=500
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 500, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        with pytest.raises(ValueError, match="different datagen runs"):
            SyntheticFileList.peek_total_from_disk(
                data_folder=str(tmp_path), dataset_type="train",
                num_subfolders=0, file_prefix="img", file_format="bin")

    def test_peek_mixed_series_with_expected_total_disambiguates(self, tmp_path):
        (tmp_path / "train").mkdir()
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 100, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 500, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        # User asserts they want the N=500 series
        n = SyntheticFileList.peek_total_from_disk(
            data_folder=str(tmp_path), dataset_type="train",
            num_subfolders=0, file_prefix="img", file_format="bin",
            expected_total=500)
        assert n == 500
        # Same dir, opposite series
        n = SyntheticFileList.peek_total_from_disk(
            data_folder=str(tmp_path), dataset_type="train",
            num_subfolders=0, file_prefix="img", file_format="bin",
            expected_total=100)
        assert n == 100

    def test_peek_mixed_series_with_wrong_expected_total_raises(self, tmp_path):
        (tmp_path / "train").mkdir()
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 100, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 500, 0, file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            SyntheticFileList.peek_total_from_disk(
                data_folder=str(tmp_path), dataset_type="train",
                num_subfolders=0, file_prefix="img", file_format="bin",
                expected_total=999)  # not one of the actual Ns


class TestSyntheticFileListEnumLikeInputs:
    """DLIO passes config values as enum objects (DatasetType.TRAIN,
    FormatType.JPEG) whose str() matches their value. We must coerce."""

    class _FakeFormat:
        def __init__(self, v): self.v = v
        def __str__(self): return self.v
        def __repr__(self): return self.v

    def test_init_with_enum_like_inputs(self, tmp_path):
        fmt = self._FakeFormat("bin")
        ds = self._FakeFormat("train")
        prefix = self._FakeFormat("img")
        (tmp_path / "train").mkdir()
        for i in range(5):
            p = _datagen_path(str(tmp_path), "train", i, 5, 0,
                              file_prefix="img", file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        sfl = SyntheticFileList(str(tmp_path), ds, 5, 0,
                                file_prefix=prefix, file_format=fmt)
        assert sfl[0].endswith("_of_5.bin")
        sfl.validate_on_disk(num_random_probes=2)

    def test_peek_with_enum_like_inputs(self, tmp_path):
        fmt = self._FakeFormat("bin")
        ds = self._FakeFormat("train")
        prefix = self._FakeFormat("img")
        (tmp_path / "train").mkdir()
        for i in range(3):
            p = _datagen_path(str(tmp_path), "train", i, 3, 0,
                              file_prefix="img", file_format="bin")
            with open(p, "wb") as f:
                f.write(b"\x00")
        n = SyntheticFileList.peek_total_from_disk(
            data_folder=str(tmp_path), dataset_type=ds, num_subfolders=0,
            file_prefix=prefix, file_format=fmt)
        assert n == 3


# ---------------------------------------------------------------------------
# Integration: SyntheticFileList → VirtualIndexMap (the real call path)
# ---------------------------------------------------------------------------

class TestSyntheticFileListWithVirtualIndexMap:
    """Make sure VirtualIndexMap can consume the lazy file list seamlessly."""

    def test_virtual_index_map_resolves_synthetic_paths(self):
        # Load VirtualIndexMap the same way the existing test does
        sys.path.insert(0, os.path.dirname(__file__))
        from test_virtual_index_map import VirtualIndexMap

        n = 1000
        spf = 4
        sfl = SyntheticFileList("/data/root", "train", n, 10, file_format="jpeg")
        vmap = VirtualIndexMap(sfl, num_samples_per_file=spf,
                               start_sample=0, end_sample=spf * n - 1,
                               storage_type="local_fs")
        # For absolute roots the abspath prefix never fires, so the path
        # must equal what SyntheticFileList produces.
        for s in [0, spf - 1, spf, spf + 1, spf * n - 1]:
            path, off = vmap[s]
            assert path == sfl[s // spf]
            assert off == s % spf

    def test_virtual_index_map_relative_path_gets_abspath_prefix(self):
        """When data_folder is relative we must compose against cwd."""
        sys.path.insert(0, os.path.dirname(__file__))
        from test_virtual_index_map import VirtualIndexMap

        sfl = SyntheticFileList("rel/root", "train", 10, 0, file_format="npz")
        vmap = VirtualIndexMap(sfl, num_samples_per_file=1,
                               start_sample=0, end_sample=9,
                               storage_type="local_fs")
        path, _ = vmap[0]
        # Resolved path should be absolute now
        assert os.path.isabs(path), f"expected absolute path, got {path!r}"
        assert path.endswith(sfl[0])

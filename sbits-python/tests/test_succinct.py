import numpy as np
import pytest

from sbits import BitVector, EliasFano, PartitionedEliasFano, WaveletTree


def test_bitvector_rank():
    bv = BitVector([True, False, True, True, False, True])
    assert len(bv) == 6
    assert bv.rank(0) == 0
    assert bv.rank(1) == 1
    assert bv.rank(3) == 2
    assert bv.rank(4) == 3
    assert bv.rank(6) == 4


def test_bitvector_select():
    bv = BitVector([True, False, True, True, False, True])
    assert bv.select(0) == 0  # 1st set bit at position 0
    assert bv.select(1) == 2  # 2nd set bit at position 2
    assert bv.select(2) == 3  # 3rd set bit at position 3
    assert bv.select(3) == 5  # 4th set bit at position 5
    assert bv.select(4) is None


def test_bitvector_getitem():
    bv = BitVector([True, False, True])
    assert bv[0] is True
    assert bv[1] is False
    assert bv[2] is True
    assert bv[-1] is True
    assert bv[-2] is False


def test_bitvector_count_ones():
    bv = BitVector([True, False, True, True, False, True])
    assert bv.count_ones() == 4


def test_bitvector_repr():
    bv = BitVector([True, False, True])
    assert "len=3" in repr(bv)
    assert "ones=2" in repr(bv)


def test_elias_fano_roundtrip():
    values = [10, 20, 30, 100, 1000]
    ef = EliasFano(values)
    assert len(ef) == 5
    for i, v in enumerate(values):
        assert ef.get(i) == v
        assert ef[i] == v


def test_elias_fano_contains():
    ef = EliasFano([10, 20, 30, 100, 1000])
    assert 10 in ef
    assert 30 in ef
    assert 1000 in ef
    assert 15 not in ef
    assert 0 not in ef


def test_elias_fano_negative_index():
    ef = EliasFano([10, 20, 30])
    assert ef[-1] == 30
    assert ef[-3] == 10


def test_wavelet_tree_rank_select():
    data = [3, 1, 2, 0, 3, 0, 1, 2]
    wt = WaveletTree(data, sigma=4)
    assert len(wt) == 8

    # access
    assert wt.access(0) == 3
    assert wt.access(3) == 0

    # rank: count of symbol in [0, i)
    assert wt.rank(3, 8) == 2
    assert wt.rank(0, 8) == 2
    assert wt.rank(3, 1) == 1
    assert wt.rank(3, 0) == 0

    # select: position of k-th occurrence (0-indexed)
    assert wt.select(3, 0) == 0
    assert wt.select(3, 1) == 4
    assert wt.select(0, 0) == 3
    assert wt.select(3, 2) is None


def test_empty_bitvector():
    bv = BitVector([])
    assert len(bv) == 0
    assert bv.count_ones() == 0
    assert bv.rank(0) == 0
    assert bv.select(0) is None


# -- numpy integration -------------------------------------------------------


def test_bitvector_from_numpy():
    bits = np.array([True, False, True, True, False], dtype=np.bool_)
    bv = BitVector(bits)
    assert len(bv) == 5
    assert bv.rank(3) == 2
    assert bv[0] is True
    assert bv[1] is False


def test_bitvector_to_numpy():
    bv = BitVector([True, False, True])
    arr = bv.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.bool_
    assert list(arr) == [True, False, True]


def test_bitvector_numpy_roundtrip():
    original = np.array([True, False, True, True, False, True], dtype=np.bool_)
    bv = BitVector(original)
    result = bv.to_numpy()
    np.testing.assert_array_equal(original, result)


def test_elias_fano_from_numpy_uint32():
    values = np.array([10, 20, 30, 100], dtype=np.uint32)
    ef = EliasFano(values)
    assert len(ef) == 4
    assert ef[0] == 10
    assert ef[3] == 100


def test_elias_fano_from_numpy_int64():
    values = np.array([10, 20, 30, 100], dtype=np.int64)
    ef = EliasFano(values)
    assert ef[0] == 10
    assert ef[3] == 100


def test_elias_fano_from_numpy_uint64_above_u32():
    values = np.array([10, 2**32 + 17], dtype=np.uint64)
    ef = EliasFano(values)
    assert ef[1] == 2**32 + 17
    assert ef.successor(2**32) == 2**32 + 17
    assert ef.predecessor(2**32 + 18) == 2**32 + 17


def test_elias_fano_to_numpy():
    ef = EliasFano([10, 20, 30, 100])
    arr = ef.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint64
    assert list(arr) == [10, 20, 30, 100]


def test_elias_fano_rejects_unrepresentable_universe():
    with pytest.raises(OverflowError, match="universe would overflow"):
        EliasFano([2**64 - 1])


def test_wavelet_tree_from_numpy():
    data = np.array([3, 1, 2, 0, 3, 0, 1, 2], dtype=np.uint32)
    wt = WaveletTree(data, sigma=4)
    assert len(wt) == 8
    assert wt.access(0) == 3
    assert wt.rank(3, 8) == 2


def test_wavelet_tree_getitem():
    data = [3, 1, 2, 0, 3, 0, 1, 2]
    wt = WaveletTree(data, sigma=4)
    for i, v in enumerate(data):
        assert wt[i] == v
    # Negative indexing.
    assert wt[-1] == 2
    assert wt[-8] == 3


def test_bitvector_iter():
    bits = [True, False, True, True, False]
    bv = BitVector(bits)
    assert list(bv) == bits


# -- rank0 / select0 --------------------------------------------------------


def test_bitvector_rank0():
    # bits: T F T T F T -> zeros at positions 1 and 4
    bv = BitVector([True, False, True, True, False, True])
    assert bv.rank0(0) == 0
    assert bv.rank0(1) == 0  # bit 0 is set -> 0 zeros in [0,1)
    assert bv.rank0(2) == 1  # bit 1 is unset -> 1 zero in [0,2)
    assert bv.rank0(5) == 2  # zeros at positions 1 and 4 -> 2 zeros in [0,5)
    assert bv.rank0(6) == 2


def test_bitvector_select0():
    bv = BitVector([True, False, True, True, False, True])
    # zeros at positions 1 and 4
    assert bv.select0(0) == 1
    assert bv.select0(1) == 4
    assert bv.select0(2) is None


# -- BitVector serialization -------------------------------------------------


def test_bitvector_serialization_roundtrip():
    bits = [True, False, True, True, False, True, False, False, True]
    bv = BitVector(bits)
    data = bv.to_bytes()
    assert isinstance(data, bytes)
    bv2 = BitVector.from_bytes(data)
    assert len(bv2) == len(bv)
    assert bv2.count_ones() == bv.count_ones()
    for i in range(len(bv)):
        assert bv2[i] == bv[i]


def test_bitvector_from_bytes_rejects_bad_data():
    with pytest.raises(ValueError):
        BitVector.from_bytes(b"not valid data")


# -- EliasFano serialization -------------------------------------------------


def test_elias_fano_serialization_roundtrip():
    values = [10, 20, 30, 100, 1000]
    ef = EliasFano(values)
    data = ef.to_bytes()
    assert isinstance(data, bytes)
    ef2 = EliasFano.from_bytes(data)
    assert len(ef2) == len(ef)
    for i in range(len(ef)):
        assert ef2[i] == ef[i]


def test_elias_fano_from_bytes_rejects_bad_data():
    with pytest.raises(ValueError):
        EliasFano.from_bytes(b"not valid data")


# -- PartitionedEliasFano ----------------------------------------------------


def test_partitioned_elias_fano_basic():
    values = [10, 20, 30, 31, 32, 100, 1000]
    pef = PartitionedEliasFano(values, block_size=3)
    assert len(pef) == len(values)
    for i, v in enumerate(values):
        assert pef.get(i) == v
        assert pef[i] == v


def test_partitioned_elias_fano_negative_index():
    pef = PartitionedEliasFano([10, 20, 30])
    assert pef[-1] == 30
    assert pef[-3] == 10


def test_partitioned_elias_fano_repr():
    pef = PartitionedEliasFano([10, 20, 30])
    assert "len=3" in repr(pef)


def test_partitioned_elias_fano_serialization_roundtrip():
    values = [10, 20, 30, 31, 32, 100, 1000]
    pef = PartitionedEliasFano(values, block_size=3)
    data = pef.to_bytes()
    assert isinstance(data, bytes)
    pef2 = PartitionedEliasFano.from_bytes(data)
    assert len(pef2) == len(pef)
    for i in range(len(pef)):
        assert pef2[i] == pef[i]


def test_partitioned_elias_fano_from_bytes_rejects_bad_data():
    with pytest.raises(ValueError):
        PartitionedEliasFano.from_bytes(b"not valid data")


def test_partitioned_elias_fano_empty():
    pef = PartitionedEliasFano([])
    assert len(pef) == 0


def test_partitioned_elias_fano_numpy():
    values = np.array([5, 10, 15, 20], dtype=np.uint32)
    pef = PartitionedEliasFano(values, block_size=2)
    assert len(pef) == 4
    assert pef[0] == 5
    assert pef[3] == 20


def test_partitioned_elias_fano_uint64_above_u32():
    values = np.array([5, 2**32 + 17], dtype=np.uint64)
    pef = PartitionedEliasFano(values, block_size=2)
    assert pef[1] == 2**32 + 17


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


def test_sbits_elias_fano_unsorted_raises():
    """EliasFano with unsorted input must raise ValueError."""
    with pytest.raises(ValueError, match="values must be sorted"):
        EliasFano([10, 5, 20])
    with pytest.raises(ValueError, match="values must be sorted"):
        EliasFano([1, 3, 2, 4])

import numpy as np

from sbits import BitVector, EliasFano, WaveletTree


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


def test_elias_fano_to_numpy():
    ef = EliasFano([10, 20, 30, 100])
    arr = ef.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint32
    assert list(arr) == [10, 20, 30, 100]


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

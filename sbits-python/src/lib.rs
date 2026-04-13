use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

// ---------------------------------------------------------------------------
// BitVector
// ---------------------------------------------------------------------------

/// Succinct bit vector with O(1) rank and O(log n) select.
///
/// Space: n + o(n) bits.
///
/// Construct from a list of bools or a numpy bool array:
///
///     bv = BitVector([True, False, True, True, False, True])
///     bv.rank(4)   # 3
///     bv.select(2) # 3  (position of 3rd set bit, 0-indexed)
#[pyclass]
struct BitVector {
    inner: sbits_core::BitVector,
}

impl BitVector {
    fn from_bools(bools: &[bool]) -> Self {
        let len = bools.len();
        let num_words = len.div_ceil(64);
        let mut words = vec![0u64; num_words];
        for (i, &b) in bools.iter().enumerate() {
            if b {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self {
            inner: sbits_core::BitVector::new(&words, len),
        }
    }
}

#[pymethods]
impl BitVector {
    /// Construct from a sequence of booleans (list[bool] or numpy bool array).
    #[new]
    fn new(bits: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bools: Vec<bool> = if let Ok(arr) = bits.extract::<PyReadonlyArray1<bool>>() {
            arr.as_array().to_vec()
        } else {
            bits.extract::<Vec<bool>>()?
        };
        Ok(Self::from_bools(&bools))
    }

    /// Count set bits in [0, i). O(1) time.
    fn rank(&self, i: usize) -> usize {
        self.inner.rank1(i)
    }

    /// Count unset bits in [0, i). O(1) time.
    fn rank0(&self, i: usize) -> usize {
        self.inner.rank0(i)
    }

    /// Position of the k-th set bit (0-indexed). O(log n) time.
    ///
    /// Returns None if k >= count_ones.
    fn select(&self, k: usize) -> Option<usize> {
        self.inner.select1(k)
    }

    /// Position of the k-th unset bit (0-indexed). O(log n) time.
    ///
    /// Returns None if k >= count_zeros.
    fn select0(&self, k: usize) -> Option<usize> {
        self.inner.select0(k)
    }

    /// Return True if the bit at index i is set.
    fn get(&self, i: usize) -> bool {
        self.inner.get(i)
    }

    /// Number of set bits in the entire vector.
    fn count_ones(&self) -> usize {
        self.inner.rank1(self.inner.len())
    }

    /// Export all bits as a numpy boolean array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let len = self.inner.len();
        let bools: Vec<bool> = (0..len).map(|i| self.inner.get(i)).collect();
        PyArray1::from_vec(py, bools)
    }

    /// Serialize to bytes for persistence or pickle support.
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bytes produced by ``to_bytes()``.
    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let inner = sbits_core::BitVector::from_bytes(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<bool> {
        let len = self.inner.len() as isize;
        let idx = if i < 0 { len + i } else { i };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.get(idx as usize))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> BitVectorIter {
        BitVectorIter {
            inner: slf.inner.clone(),
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        let ones = self.inner.rank1(self.inner.len());
        format!("BitVector(len={}, ones={})", self.inner.len(), ones)
    }
}

#[pyclass]
struct BitVectorIter {
    inner: sbits_core::BitVector,
    index: usize,
}

#[pymethods]
impl BitVectorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<bool> {
        if self.index < self.inner.len() {
            let val = self.inner.get(self.index);
            self.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// EliasFano
// ---------------------------------------------------------------------------

/// Elias-Fano encoding for sorted sequences of non-negative integers.
///
/// Space: 2n + n*ceil(log2(u/n)) + o(n) bits, where u is the universe size.
///
/// Construct from a sorted list or numpy array:
///
///     ef = EliasFano([10, 20, 30, 100, 1000])
///     ef[0]        # 10
///     100 in ef    # True
#[pyclass]
struct EliasFano {
    inner: sbits_core::EliasFano,
}

#[pymethods]
impl EliasFano {
    /// Construct from a sorted sequence (list[int] or numpy uint32/int64 array).
    ///
    /// Values must be in non-decreasing order.
    #[new]
    fn new(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vals: Vec<u32> = if let Ok(arr) = values.extract::<PyReadonlyArray1<u32>>() {
            arr.as_array().to_vec()
        } else if let Ok(arr) = values.extract::<PyReadonlyArray1<i64>>() {
            arr.as_array()
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        pyo3::exceptions::PyOverflowError::new_err(format!(
                            "value {v} out of u32 range"
                        ))
                    })
                })
                .collect::<PyResult<Vec<u32>>>()?
        } else {
            values.extract::<Vec<u32>>()?
        };
        for i in 1..vals.len() {
            if vals[i] < vals[i - 1] {
                return Err(PyValueError::new_err(format!(
                    "values must be sorted: found {} before {} at index {}",
                    vals[i - 1],
                    vals[i],
                    i
                )));
            }
        }
        let universe = vals.last().map(|&v| v + 1).unwrap_or(0);
        Ok(Self {
            inner: sbits_core::EliasFano::new(&vals, universe),
        })
    }

    /// Return the value at index i. O(1) time.
    fn get(&self, i: usize) -> PyResult<u32> {
        self.inner
            .get(i)
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    /// Return True if the value is present. O(log n) time.
    fn contains(&self, value: u32) -> bool {
        self.inner.contains(value)
    }

    /// Smallest value >= target, or None.
    fn successor(&self, target: u32) -> Option<u32> {
        self.inner.successor(target)
    }

    /// Largest value <= target, or None.
    fn predecessor(&self, target: u32) -> Option<u32> {
        self.inner.predecessor(target)
    }

    /// Alias for successor (standard IR name).
    fn next_geq(&self, target: u32) -> Option<u32> {
        self.inner.next_geq(target)
    }

    /// Export all values as a numpy uint32 array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let n = self.inner.len();
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            vals.push(
                self.inner
                    .get(i)
                    .map_err(|e| PyIndexError::new_err(e.to_string()))?,
            );
        }
        Ok(PyArray1::from_vec(py, vals))
    }

    /// Serialize to bytes for persistence or pickle support.
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bytes produced by ``to_bytes()``.
    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let inner = sbits_core::EliasFano::from_bytes(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, value: u32) -> bool {
        self.contains(value)
    }

    fn __getitem__(&self, i: isize) -> PyResult<u32> {
        let len = self.inner.len() as isize;
        let idx = if i < 0 { len + i } else { i };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("index out of range"));
        }
        self.inner
            .get(idx as usize)
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("EliasFano(len={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// PartitionedEliasFano
// ---------------------------------------------------------------------------

/// Partitioned Elias-Fano encoding for clustered monotone sequences.
///
/// Splits the sequence into blocks and encodes each with its own local
/// universe, improving compression for clustered data.
///
/// Construct from a sorted list or numpy array:
///
///     pef = PartitionedEliasFano([10, 20, 30, 31, 32, 100, 1000], block_size=3)
///     pef[0]  # 10
///     len(pef) # 7
#[pyclass]
struct PartitionedEliasFano {
    inner: sbits_core::PartitionedEliasFano,
}

#[pymethods]
impl PartitionedEliasFano {
    /// Construct from a sorted sequence and block size.
    ///
    /// Args:
    ///     values: Sorted values as numpy array (uint32 or int64) or list.
    ///     block_size: Maximum number of items per block (default 128).
    #[new]
    #[pyo3(signature = (values, block_size=128))]
    fn new(values: &Bound<'_, PyAny>, block_size: usize) -> PyResult<Self> {
        let vals: Vec<u32> = if let Ok(arr) = values.extract::<PyReadonlyArray1<u32>>() {
            arr.as_array().to_vec()
        } else if let Ok(arr) = values.extract::<PyReadonlyArray1<i64>>() {
            arr.as_array()
                .iter()
                .map(|&v| {
                    u32::try_from(v).map_err(|_| {
                        pyo3::exceptions::PyOverflowError::new_err(format!(
                            "value {v} out of u32 range"
                        ))
                    })
                })
                .collect::<PyResult<Vec<u32>>>()?
        } else {
            values.extract::<Vec<u32>>()?
        };
        let universe = vals.last().map(|&v| v + 1).unwrap_or(0);
        Ok(Self {
            inner: sbits_core::PartitionedEliasFano::new(&vals, universe, block_size),
        })
    }

    /// Return the value at index i. O(1) time.
    fn get(&self, i: usize) -> PyResult<u32> {
        self.inner
            .get(i)
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    /// Serialize to bytes for persistence or pickle support.
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bytes produced by ``to_bytes()``.
    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let inner = sbits_core::PartitionedEliasFano::from_bytes(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<u32> {
        let len = self.inner.len() as isize;
        let idx = if i < 0 { len + i } else { i };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("index out of range"));
        }
        self.inner
            .get(idx as usize)
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("PartitionedEliasFano(len={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// WaveletTree
// ---------------------------------------------------------------------------

/// Wavelet tree for rank/select over arbitrary integer alphabets.
///
/// Space: n*ceil(log2(sigma)) + o(n*log(sigma)) bits.
///
/// Construct from a list or numpy array and alphabet size:
///
///     wt = WaveletTree([3, 1, 2, 0, 3, 0, 1, 2], sigma=4)
///     wt.access(0)      # 3
///     wt.rank(3, 8)     # 2
///     wt.select(3, 0)   # 0
#[pyclass]
struct WaveletTree {
    inner: sbits_core::WaveletTree,
}

#[pymethods]
impl WaveletTree {
    /// Construct from a sequence and alphabet size (list[int] or numpy uint32 array).
    #[new]
    fn new(data: &Bound<'_, PyAny>, sigma: u32) -> PyResult<Self> {
        let vals: Vec<u32> = if let Ok(arr) = data.extract::<PyReadonlyArray1<u32>>() {
            arr.as_array().to_vec()
        } else {
            data.extract::<Vec<u32>>()?
        };
        Ok(Self {
            inner: sbits_core::WaveletTree::new(&vals, sigma),
        })
    }

    /// Count occurrences of symbol in [0, i). O(log(sigma)) time.
    fn rank(&self, symbol: u32, i: usize) -> usize {
        self.inner.rank(symbol, i)
    }

    /// Position of the k-th occurrence of symbol (0-indexed). O(log(sigma)) time.
    ///
    /// Returns None if fewer than k+1 occurrences exist.
    fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        self.inner.select(symbol, k)
    }

    /// Return the symbol at index i. O(log(sigma)) time.
    fn access(&self, i: usize) -> u32 {
        self.inner.access(i)
    }

    /// Serialize to bytes for persistence or pickle support.
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bytes produced by ``to_bytes()``.
    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let inner = sbits_core::WaveletTree::from_bytes(data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<u32> {
        let len = self.inner.len() as isize;
        let idx = if i < 0 { len + i } else { i };
        if idx < 0 || idx >= len {
            return Err(PyIndexError::new_err("index out of range"));
        }
        Ok(self.inner.access(idx as usize))
    }

    fn __repr__(&self) -> String {
        format!("WaveletTree(len={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn sbits(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.3")?;
    m.add_class::<BitVector>()?;
    m.add_class::<EliasFano>()?;
    m.add_class::<PartitionedEliasFano>()?;
    m.add_class::<WaveletTree>()?;
    Ok(())
}

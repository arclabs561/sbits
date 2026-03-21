use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// BitVector
// ---------------------------------------------------------------------------

/// Succinct bit vector with O(1) rank and select.
///
/// Construct from a list of bools:
///
///     bv = BitVector([True, False, True, True, False, True])
///     bv.rank(4)   # 3
///     bv.select(2) # 3  (position of 3rd set bit, 0-indexed)
#[pyclass]
struct BitVector {
    inner: sbits_core::BitVector,
}

#[pymethods]
impl BitVector {
    #[new]
    fn new(bits: Vec<bool>) -> Self {
        let len = bits.len();
        let num_words = len.div_ceil(64);
        let mut words = vec![0u64; num_words];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self {
            inner: sbits_core::BitVector::new(&words, len),
        }
    }

    /// Number of set bits in [0, i).
    fn rank(&self, i: usize) -> usize {
        self.inner.rank1(i)
    }

    /// Position of the k-th set bit (0-indexed). Returns None if k >= count_ones.
    fn select(&self, k: usize) -> Option<usize> {
        self.inner.select1(k)
    }

    /// Return True if the bit at index i is set.
    fn get(&self, i: usize) -> bool {
        self.inner.get(i)
    }

    /// Number of set bits in the entire vector.
    fn count_ones(&self) -> usize {
        self.inner.rank1(self.inner.len())
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

    fn __repr__(&self) -> String {
        let ones = self.inner.rank1(self.inner.len());
        format!("BitVector(len={}, ones={})", self.inner.len(), ones)
    }
}

// ---------------------------------------------------------------------------
// EliasFano
// ---------------------------------------------------------------------------

/// Elias-Fano encoding for sorted sequences of non-negative integers.
///
/// Construct from a sorted list:
///
///     ef = EliasFano([10, 20, 30, 100, 1000])
///     ef[0]        # 10
///     100 in ef    # True (linear scan)
#[pyclass]
struct EliasFano {
    inner: sbits_core::EliasFano,
}

#[pymethods]
impl EliasFano {
    #[new]
    fn new(values: Vec<u32>) -> PyResult<Self> {
        let universe = values.last().map(|&v| v + 1).unwrap_or(0);
        Ok(Self {
            inner: sbits_core::EliasFano::new(&values, universe),
        })
    }

    /// Return the value at index i.
    fn get(&self, i: usize) -> PyResult<u32> {
        self.inner
            .get(i)
            .map_err(|e| PyIndexError::new_err(e.to_string()))
    }

    /// Return True if the value is present (linear scan).
    fn contains(&self, value: u32) -> PyResult<bool> {
        for i in 0..self.inner.len() {
            let v = self
                .inner
                .get(i)
                .map_err(|e| PyIndexError::new_err(e.to_string()))?;
            if v == value {
                return Ok(true);
            }
            if v > value {
                return Ok(false);
            }
        }
        Ok(false)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __contains__(&self, value: u32) -> PyResult<bool> {
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
// WaveletTree
// ---------------------------------------------------------------------------

/// Wavelet tree for rank/select over arbitrary integer alphabets.
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
    #[new]
    fn new(data: Vec<u32>, sigma: u32) -> Self {
        Self {
            inner: sbits_core::WaveletTree::new(&data, sigma),
        }
    }

    /// Number of occurrences of symbol in [0, i).
    fn rank(&self, symbol: u32, i: usize) -> usize {
        self.inner.rank(symbol, i)
    }

    /// Position of the k-th occurrence of symbol (0-indexed).
    fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        self.inner.select(symbol, k)
    }

    /// Return the symbol at index i.
    fn access(&self, i: usize) -> u32 {
        self.inner.access(i)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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
    m.add_class::<BitVector>()?;
    m.add_class::<EliasFano>()?;
    m.add_class::<WaveletTree>()?;
    Ok(())
}

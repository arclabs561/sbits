//! Wavelet Matrix for arbitrary alphabets.
//!
//! Generalizes rank and select operations from bit vectors to
//! sequences over larger alphabets. Uses a flat matrix layout
//! (one bitvector per bit-level) instead of a recursive tree,
//! eliminating pointer chasing for better cache performance.
//!
//! Total space: `n * ceil(log2(sigma)) + o(n * log(sigma))` bits.
//! Queries `access`, `rank`, `select` take `O(log(sigma))` time.

use crate::bitvec::BitVector;
use crate::error::{ByteReader, Result};

/// Wavelet Matrix over an integer alphabet.
///
/// Internally stores one [`BitVector`] per bit-level of the alphabet,
/// with a "zeros boundary" per level separating left (0-bit) from right (1-bit)
/// elements. This flat layout avoids the heap-allocated tree nodes of a
/// traditional wavelet tree.
#[derive(Debug, Clone)]
pub struct WaveletTree {
    /// One bitvector per bit-level, from MSB (index 0) to LSB.
    levels: Vec<BitVector>,
    /// Number of zeros at each level (boundary between left/right halves).
    zeros: Vec<usize>,
    len: usize,
    sigma: u32,
    /// Number of bit-levels: ceil(log2(sigma)), or 0 if sigma <= 1.
    depth: usize,
}

impl WaveletTree {
    /// Create a new Wavelet Matrix from a sequence of symbols.
    ///
    /// # Panics
    ///
    /// Panics if any symbol in `data` is >= `sigma`.
    pub fn new(data: &[u32], sigma: u32) -> Self {
        for (i, &v) in data.iter().enumerate() {
            assert!(
                v < sigma,
                "WaveletTree: symbol {} at index {} >= sigma {}",
                v,
                i,
                sigma
            );
        }

        let depth = if sigma <= 1 {
            0
        } else {
            (u32::BITS - (sigma - 1).leading_zeros()) as usize
        };

        let n = data.len();
        let mut levels = Vec::with_capacity(depth);
        let mut zeros = Vec::with_capacity(depth);

        // Working copy of the sequence, reordered at each level.
        let mut current: Vec<u32> = data.to_vec();

        for level in 0..depth {
            let bit_pos = depth - 1 - level; // MSB first
            let mut bits = vec![0u64; n.div_ceil(64)];
            let mut left = Vec::new();
            let mut right = Vec::new();

            for (i, &v) in current.iter().enumerate() {
                if (v >> bit_pos) & 1 == 1 {
                    bits[i / 64] |= 1u64 << (i % 64);
                    right.push(v);
                } else {
                    left.push(v);
                }
            }

            let bv = BitVector::new(&bits, n);
            zeros.push(bv.rank0(n));
            levels.push(bv);

            // Stable partition: all 0-bit elements, then all 1-bit elements.
            current.clear();
            current.extend_from_slice(&left);
            current.extend_from_slice(&right);
        }

        Self {
            levels,
            zeros,
            len: n,
            sigma,
            depth,
        }
    }

    /// Return the alphabet size.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }

    /// Return the length of the sequence.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true if the sequence has length 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the symbol at index `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= len`. Use [`get`](Self::get) for a non-panicking version.
    pub fn access(&self, mut i: usize) -> u32 {
        assert!(
            i < self.len,
            "WaveletTree::access: index {i} >= len {}",
            self.len
        );
        let mut symbol = 0u32;
        for level in 0..self.depth {
            let bit_pos = self.depth - 1 - level;
            if self.levels[level].get(i) {
                // Bit is 1: go right.
                symbol |= 1 << bit_pos;
                i = self.zeros[level] + self.levels[level].rank1(i);
            } else {
                // Bit is 0: go left.
                i = self.levels[level].rank0(i);
            }
        }
        symbol
    }

    /// Return the number of occurrences of `symbol` in the range \[0, i).
    pub fn rank(&self, symbol: u32, mut i: usize) -> usize {
        // Track the start of the symbol's region alongside i to avoid a second traversal.
        let mut start = 0usize;
        for level in 0..self.depth {
            let bit_pos = self.depth - 1 - level;
            if (symbol >> bit_pos) & 1 == 1 {
                start = self.zeros[level] + self.levels[level].rank1(start);
                i = self.zeros[level] + self.levels[level].rank1(i);
            } else {
                start = self.levels[level].rank0(start);
                i = self.levels[level].rank0(i);
            }
        }
        i - start
    }

    /// Return the position of the `k`-th occurrence of `symbol` (0-indexed).
    pub fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        // Find the position in the bottom level, then map back up.
        let start = self.symbol_start(symbol);
        let mut i = start + k;

        // Walk back up from the bottom level.
        for level in (0..self.depth).rev() {
            let bit_pos = self.depth - 1 - level;
            if (symbol >> bit_pos) & 1 == 1 {
                // This position was in the right half (1-bit region).
                let rank_in_right = i - self.zeros[level];
                i = self.levels[level].select1(rank_in_right)?;
            } else {
                // This position was in the left half (0-bit region).
                i = self.levels[level].select0(i)?;
            }
        }
        if i < self.len {
            Some(i)
        } else {
            None
        }
    }

    /// Return the symbol at index `i`, or `None` if out of bounds.
    pub fn get(&self, i: usize) -> Option<u32> {
        if i < self.len {
            Some(self.access(i))
        } else {
            None
        }
    }

    /// Heap memory usage in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.levels.iter().map(|bv| bv.heap_bytes()).sum::<usize>()
            + self.zeros.len() * std::mem::size_of::<usize>()
    }

    /// Serialize this wavelet matrix to a stable binary encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITWM01");
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&self.sigma.to_le_bytes());
        out.extend_from_slice(&(self.depth as u32).to_le_bytes());
        for level in &self.levels {
            let bv_bytes = level.to_bytes();
            out.extend_from_slice(&(bv_bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bv_bytes);
        }
        for &z in &self.zeros {
            out.extend_from_slice(&(z as u64).to_le_bytes());
        }
        out
    }

    /// Deserialize a wavelet matrix from `to_bytes()` output.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(bytes);
        r.read_magic(b"SBITWM01", "WaveletMatrix")?;
        let len = r.read_u64()? as usize;
        let sigma = r.read_u32()?;
        let depth = r.read_u32()? as usize;

        let mut levels = Vec::with_capacity(depth);
        for _ in 0..depth {
            let bv_len = r.read_u64()? as usize;
            let bv_bytes = r.take(bv_len)?;
            levels.push(BitVector::from_bytes(bv_bytes)?);
        }

        let mut zeros = Vec::with_capacity(depth);
        for _ in 0..depth {
            zeros.push(r.read_u64()? as usize);
        }

        r.expect_eof("WaveletMatrix")?;
        Ok(Self {
            levels,
            zeros,
            len,
            sigma,
            depth,
        })
    }

    /// Compute the start position of a symbol's region in the bottom level.
    fn symbol_start(&self, symbol: u32) -> usize {
        let mut lo = 0usize;
        let mut hi = self.len;
        for level in 0..self.depth {
            let bit_pos = self.depth - 1 - level;
            if (symbol >> bit_pos) & 1 == 1 {
                lo = self.zeros[level] + self.levels[level].rank1(lo);
                hi = self.zeros[level] + self.levels[level].rank1(hi);
            } else {
                lo = self.levels[level].rank0(lo);
                hi = self.levels[level].rank0(hi);
            }
        }
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_tree_basic() {
        let data = vec![3, 1, 2, 0, 3, 0, 1, 2];
        let wt = WaveletTree::new(&data, 4);

        assert_eq!(wt.len(), 8);
        assert_eq!(wt.access(0), 3);
        assert_eq!(wt.access(3), 0);

        assert_eq!(wt.rank(3, 8), 2);
        assert_eq!(wt.rank(0, 8), 2);
        assert_eq!(wt.rank(1, 8), 2);
        assert_eq!(wt.rank(2, 8), 2);

        assert_eq!(wt.rank(3, 4), 1);
        assert_eq!(wt.rank(0, 4), 1);
    }

    #[test]
    fn test_wavelet_tree_select() {
        let data = vec![3, 1, 2, 0, 3, 0, 1, 2];
        let wt = WaveletTree::new(&data, 4);

        assert_eq!(wt.select(3, 0), Some(0));
        assert_eq!(wt.select(3, 1), Some(4));
        assert_eq!(wt.select(0, 0), Some(3));
        assert_eq!(wt.select(0, 1), Some(5));
        assert_eq!(wt.select(2, 1), Some(7));
        assert_eq!(wt.select(3, 2), None);
        assert_eq!(wt.select(0, 2), None);
        assert_eq!(wt.select(1, 2), None);
        assert_eq!(wt.select(2, 2), None);
    }

    #[test]
    fn test_wavelet_tree_sigma_1() {
        let data = vec![0, 0, 0, 0];
        let wt = WaveletTree::new(&data, 1);
        assert_eq!(wt.len(), 4);
        assert_eq!(wt.access(0), 0);
        assert_eq!(wt.access(3), 0);
        assert_eq!(wt.rank(0, 4), 4);
        assert_eq!(wt.select(0, 0), Some(0));
        assert_eq!(wt.select(0, 3), Some(3));
        assert_eq!(wt.select(0, 4), None);
    }

    #[test]
    fn test_wavelet_tree_sigma_2() {
        let data = vec![0, 1, 0, 1, 1];
        let wt = WaveletTree::new(&data, 2);
        assert_eq!(wt.rank(0, 5), 2);
        assert_eq!(wt.rank(1, 5), 3);
        assert_eq!(wt.select(0, 0), Some(0));
        assert_eq!(wt.select(0, 1), Some(2));
        assert_eq!(wt.select(1, 0), Some(1));
        assert_eq!(wt.select(1, 2), Some(4));
    }

    #[test]
    fn test_wavelet_tree_access_all() {
        let data = vec![3, 1, 2, 0, 3, 0, 1, 2];
        let wt = WaveletTree::new(&data, 4);
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(wt.access(i), expected);
        }
    }

    #[test]
    fn test_wavelet_tree_distinct_ranks() {
        let data = vec![0, 0, 0, 1, 1, 2];
        let wt = WaveletTree::new(&data, 3);
        assert_eq!(wt.rank(0, 6), 3);
        assert_eq!(wt.rank(1, 6), 2);
        assert_eq!(wt.rank(2, 6), 1);
    }

    #[test]
    fn test_wavelet_matrix_serialization() {
        let data = vec![3, 1, 2, 0, 3, 0, 1, 2];
        let wt = WaveletTree::new(&data, 4);
        let bytes = wt.to_bytes();
        let wt2 = WaveletTree::from_bytes(&bytes).unwrap();
        assert_eq!(wt2.len(), wt.len());
        assert_eq!(wt2.sigma(), wt.sigma());
        for i in 0..wt.len() {
            assert_eq!(wt2.access(i), wt.access(i));
        }
    }
}

//! Implicit data structures: near-zero space overhead.
//!
//! An "implicit" structure for a class $C_n$ uses $\log |C_n| + O(1)$ bits.
//! They represent the absolute theoretical baseline: zero indexing overhead.
//!
//! # Historical Context
//!
//! - Munro (1986): "An implicit data structure for the dictionary problem."
//! - Jacobson (1989): Succinct structures (which use $o(OPT)$ extra bits)
//!   evolved from these zero-overhead implicit structures.

/// An implicit bit vector: just the raw data with no index.
///
/// Serves as the performance baseline for `sbits::BitVector`.
#[derive(Clone, Default)]
pub struct ImplicitBitVector {
    data: Vec<u64>,
    len: usize,
}

impl ImplicitBitVector {
    /// Create a new implicit bit vector.
    pub fn new(bits: &[u64], len: usize) -> Self {
        Self {
            data: bits.to_vec(),
            len,
        }
    }

    /// Return true if bit at `i` is set. O(1).
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        (self.data[i / 64] & (1u64 << (i % 64))) != 0
    }

    /// Linear-time rank: O(N).
    pub fn rank1(&self, i: usize) -> usize {
        let i = i.min(self.len);
        let full_words = i / 64;
        let mut count = 0;
        for j in 0..full_words {
            count += self.data[j].count_ones() as usize;
        }
        let bit_offset = i % 64;
        if bit_offset > 0 {
            let mask = (1u64 << bit_offset) - 1;
            count += (self.data[full_words] & mask).count_ones() as usize;
        }
        count
    }

    /// Linear-time select: O(N).
    pub fn select1(&self, mut k: usize) -> Option<usize> {
        for (i, &word) in self.data.iter().enumerate() {
            let ones = word.count_ones() as usize;
            if k < ones {
                for bit in 0..64 {
                    if (word & (1u64 << bit)) != 0 {
                        if k == 0 {
                            return Some(i * 64 + bit);
                        }
                        k -= 1;
                    }
                }
            }
            k -= ones;
        }
        None
    }
}

//! Elias-Fano encoding for monotone sequences.
//!
//! Provides near-optimal space for sorted integers while allowing
//! $O(1)$ random access to any element.
//!
//! # Theory
//!
//! For $n$ sorted integers in range $[0, U)$, Elias-Fano uses:
//! - $L = \lfloor \log_2(U/n) \rfloor$ bits for each "lower" part.
//! - A bit vector of length $n + \lceil U/2^L \rceil$ for "upper" parts.
//!
//! Total space is $n \lceil \log_2(U/n) \rceil + 2n + o(n)$ bits.

use crate::bitvec::BitVector;
use crate::error::{Error, Result};

/// Elias-Fano encoding structure.
pub struct EliasFano {
    upper_bits: BitVector,
    lower_bits: Vec<u64>,
    l: usize,
    n: usize,
}

impl EliasFano {
    /// Create a new Elias-Fano structure from a sorted sequence.
    pub fn new(values: &[u32], universe_size: u32) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                upper_bits: BitVector::new(&[], 0),
                lower_bits: Vec::new(),
                l: 0,
                n: 0,
            };
        }

        // L = floor(log2(U/n))
        let l = if n > 0 {
            let ratio = universe_size / n as u32;
            if ratio > 0 {
                (31 - ratio.leading_zeros()) as usize
            } else {
                0
            }
        } else {
            0
        };

        // Lower bits: pack n elements of L bits each
        let mut lower_bits = Vec::with_capacity(n.saturating_mul(l).div_ceil(64));
        let mut current_word = 0u64;
        let mut bit_offset = 0;

        for &v in values {
            let low = (v & ((1 << l) - 1)) as u64;
            if bit_offset + l <= 64 {
                current_word |= low << bit_offset;
                bit_offset += l;
                if bit_offset == 64 {
                    lower_bits.push(current_word);
                    current_word = 0;
                    bit_offset = 0;
                }
            } else {
                // Split across words
                let bits_in_this = 64 - bit_offset;
                current_word |= (low & ((1 << bits_in_this) - 1)) << bit_offset;
                lower_bits.push(current_word);
                current_word = low >> bits_in_this;
                bit_offset = l - bits_in_this;
            }
        }
        if bit_offset > 0 {
            lower_bits.push(current_word);
        }

        // Upper bits: n ones and U/2^L zeros
        let num_upper_vals = (universe_size >> l) as usize + 1;
        let upper_bv_len = n + num_upper_vals;
        let mut upper_data = vec![0u64; upper_bv_len.div_ceil(64)];

        for (i, &v) in values.iter().enumerate() {
            let high = (v >> l) as usize;
            let pos = high + i;
            upper_data[pos / 64] |= 1 << (pos % 64);
        }

        Self {
            upper_bits: BitVector::new(&upper_data, upper_bv_len),
            lower_bits,
            l,
            n,
        }
    }

    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return true if the sequence has 0 elements.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Return the value at index `i`.
    pub fn get(&self, i: usize) -> Result<u32> {
        if i >= self.n {
            return Err(Error::IndexOutOfBounds(i));
        }

        // 1. Get high bits from upper_bits using select1(i)
        let pos = self
            .upper_bits
            .select1(i)
            .ok_or(Error::InvalidSelection(i))?;
        let high = (pos - i) as u32;

        // 2. Get low bits from lower_bits
        let start_bit = i * self.l;
        let word_idx = start_bit / 64;
        let bit_offset = start_bit % 64;

        let mut low = self.lower_bits[word_idx] >> bit_offset;
        if bit_offset + self.l > 64 {
            let bits_from_next = bit_offset + self.l - 64;
            low |= (self.lower_bits[word_idx + 1] & ((1 << bits_from_next) - 1))
                << (self.l - bits_from_next);
        }
        low &= (1 << self.l) - 1;

        Ok((high << self.l) | (low as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elias_fano_basic() {
        let values = vec![10, 20, 30, 100, 1000];
        let ef = EliasFano::new(&values, 2000);

        assert_eq!(ef.len(), 5);
        assert_eq!(ef.get(0).unwrap(), 10);
        assert_eq!(ef.get(1).unwrap(), 20);
        assert_eq!(ef.get(2).unwrap(), 30);
        assert_eq!(ef.get(3).unwrap(), 100);
        assert_eq!(ef.get(4).unwrap(), 1000);
    }
}

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
#[derive(Debug, Clone)]
pub struct EliasFano {
    universe_size: u32,
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
                universe_size,
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
            universe_size,
            upper_bits: BitVector::new(&upper_data, upper_bv_len),
            lower_bits,
            l,
            n,
        }
    }

    /// Return the universe size used to build this structure.
    pub fn universe_size(&self) -> u32 {
        self.universe_size
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

        // 2. Get low bits from lower_bits.
        //
        // Important edge case: when `l == 0`, there is no low part and `lower_bits` is empty.
        // (This occurs for small universes or high density where U/n <= 1.)
        let low: u32 = if self.l == 0 {
            0
        } else {
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
            low as u32
        };

        Ok((high << self.l) | low)
    }

    /// Serialize this Elias–Fano structure to a stable binary encoding (little-endian).
    ///
    /// Format (versioned):
    /// - magic: 8 bytes (`SBITEF01`)
    /// - universe_size: u32
    /// - l: u32
    /// - n: u64
    /// - lower_len: u64, then `lower_len` u64 words
    /// - upper_bits: byte_len u64, then `byte_len` bytes (BitVector::to_bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITEF01");

        out.extend_from_slice(&self.universe_size.to_le_bytes());
        out.extend_from_slice(&(self.l as u32).to_le_bytes());
        out.extend_from_slice(&(self.n as u64).to_le_bytes());

        out.extend_from_slice(&(self.lower_bits.len() as u64).to_le_bytes());
        for &w in &self.lower_bits {
            out.extend_from_slice(&w.to_le_bytes());
        }

        let upper = self.upper_bits.to_bytes();
        out.extend_from_slice(&(upper.len() as u64).to_le_bytes());
        out.extend_from_slice(&upper);
        out
    }

    /// Deserialize an Elias–Fano structure from `to_bytes()` output.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        const MAGIC: &[u8; 8] = b"SBITEF01";
        let mut off = 0usize;

        let mut take = |n: usize| -> Result<&[u8]> {
            if off + n > bytes.len() {
                return Err(Error::InvalidEncoding(
                    "unexpected end of input".to_string(),
                ));
            }
            let slice = &bytes[off..off + n];
            off += n;
            Ok(slice)
        };

        let magic = take(8)?;
        if magic != MAGIC {
            return Err(Error::InvalidEncoding(
                "bad magic for EliasFano".to_string(),
            ));
        }

        let universe_size = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let l = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        if l > 31 {
            return Err(Error::InvalidEncoding(format!(
                "EliasFano l={l} exceeds maximum (31) for u32 values"
            )));
        }
        let n = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;

        let lower_len = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        // Bound allocation against total input to prevent allocation bombs.
        if lower_len.saturating_mul(8) > bytes.len() {
            return Err(Error::InvalidEncoding(format!(
                "EliasFano lower_len ({lower_len}) too large for input ({} bytes)",
                bytes.len()
            )));
        }
        let mut lower_bits = Vec::with_capacity(lower_len);
        for _ in 0..lower_len {
            let w = u64::from_le_bytes(take(8)?.try_into().unwrap());
            lower_bits.push(w);
        }

        let upper_len = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let upper_bytes = take(upper_len)?;
        let upper_bits = BitVector::from_bytes(upper_bytes)?;

        if off != bytes.len() {
            return Err(Error::InvalidEncoding(
                "trailing bytes after EliasFano".to_string(),
            ));
        }

        Ok(Self {
            universe_size,
            upper_bits,
            lower_bits,
            l,
            n,
        })
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

    #[test]
    fn test_elias_fano_l_equals_zero() {
        // l=0 when universe_size / n <= 1 (high density).
        let values = vec![0, 1, 2, 3];
        let ef = EliasFano::new(&values, 4);
        assert_eq!(ef.len(), 4);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i).unwrap(), v, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_elias_fano_serialization_roundtrip() {
        let values = vec![10, 20, 30, 100, 1000];
        let ef = EliasFano::new(&values, 2000);
        let bytes = ef.to_bytes();
        let ef2 = EliasFano::from_bytes(&bytes).unwrap();
        assert_eq!(ef2.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef2.get(i).unwrap(), v);
        }
    }

    #[test]
    fn test_elias_fano_l0_serialization_roundtrip() {
        let values = vec![0, 1, 2, 3];
        let ef = EliasFano::new(&values, 4);
        let bytes = ef.to_bytes();
        let ef2 = EliasFano::from_bytes(&bytes).unwrap();
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef2.get(i).unwrap(), v);
        }
    }

    #[test]
    fn test_elias_fano_rejects_bad_l() {
        let ef = EliasFano::new(&[10], 100);
        let mut bytes = ef.to_bytes();
        // Corrupt the `l` field (offset 12..16) to 32.
        bytes[12] = 32;
        bytes[13] = 0;
        bytes[14] = 0;
        bytes[15] = 0;
        assert!(EliasFano::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_elias_fano_empty() {
        let ef = EliasFano::new(&[], 100);
        assert!(ef.is_empty());
        assert!(ef.get(0).is_err());
        let bytes = ef.to_bytes();
        let ef2 = EliasFano::from_bytes(&bytes).unwrap();
        assert!(ef2.is_empty());
    }
}

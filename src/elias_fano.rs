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
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

/// Elias-Fano encoding structure.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EliasFano {
    universe_size: u64,
    upper_bits: BitVector,
    lower_bits: Vec<u64>,
    l: usize,
    n: usize,
    /// Number of zeros in `upper_bits` = ceil(universe_size / 2^l) + 1.
    /// Cached to avoid calling `rank0(len)` in the hot path of `rank()`.
    num_upper_zeros: usize,
}

impl EliasFano {
    /// Create a new Elias-Fano structure from a sorted sequence.
    ///
    /// ```
    /// use sbits::EliasFano;
    ///
    /// let values = vec![10u64, 20, 30, 100, 1000];
    /// let ef = EliasFano::new(&values, 2000);
    /// assert_eq!(ef.len(), 5);
    /// assert_eq!(ef.get(0).unwrap(), 10);
    /// assert_eq!(ef.get(4).unwrap(), 1000);
    /// ```
    /// # Panics
    ///
    /// Panics if `values` is not sorted in non-decreasing order, or if any value
    /// is >= `universe_size`.
    pub fn new(values: &[u64], universe_size: u64) -> Self {
        // Validate sorted and within universe.
        for i in 0..values.len() {
            assert!(
                values[i] < universe_size,
                "EliasFano: value {} at index {} >= universe_size {}",
                values[i],
                i,
                universe_size
            );
            if i > 0 {
                assert!(
                    values[i] >= values[i - 1],
                    "EliasFano: values not sorted at index {} ({} < {})",
                    i,
                    values[i],
                    values[i - 1]
                );
            }
        }

        let n = values.len();
        if n == 0 {
            return Self {
                universe_size,
                upper_bits: BitVector::new(&[], 0),
                lower_bits: Vec::new(),
                l: 0,
                n: 0,
                num_upper_zeros: 0,
            };
        }

        // L = floor(log2(U/n))
        let ratio = universe_size / n as u64;
        let l = if ratio > 0 {
            (63 - ratio.leading_zeros()) as usize
        } else {
            0
        };

        // Lower bits: pack n elements of L bits each
        let mut lower_bits = Vec::with_capacity(n.saturating_mul(l).div_ceil(64));
        let mut current_word = 0u64;
        let mut bit_offset = 0;

        for &v in values {
            let low = v & ((1u64 << l) - 1);
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

        // Upper bits: n ones and (U/2^L + 1) zeros
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
            num_upper_zeros: num_upper_vals,
        }
    }

    /// Return the universe size used to build this structure.
    pub fn universe_size(&self) -> u64 {
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
    pub fn get(&self, i: usize) -> Result<u64> {
        if i >= self.n {
            return Err(Error::IndexOutOfBounds(i));
        }

        // 1. Get high bits from upper_bits using select1(i)
        let pos = self
            .upper_bits
            .select1(i)
            .ok_or(Error::InvalidSelection(i))?;
        let high = (pos - i) as u64;

        // 2. Get low bits from lower_bits.
        let low = self.get_lower(i);

        Ok((high << self.l) | low)
    }

    /// Read only the lower `l` bits for element at index `i`, without touching the upper bitvector.
    ///
    /// When `l == 0` (high-density case where `U/n <= 1`), `lower_bits` is empty and returns 0.
    #[inline]
    fn get_lower(&self, i: usize) -> u64 {
        if self.l == 0 {
            return 0;
        }
        let start_bit = i * self.l;
        let word_idx = start_bit / 64;
        let bit_offset = start_bit % 64;

        let mut low = self.lower_bits[word_idx] >> bit_offset;
        if bit_offset + self.l > 64 {
            let bits_from_next = bit_offset + self.l - 64;
            low |= (self.lower_bits[word_idx + 1] & ((1 << bits_from_next) - 1))
                << (self.l - bits_from_next);
        }
        low & ((1 << self.l) - 1)
    }

    /// Return the smallest value >= `target`, or `None` if no such value exists.
    ///
    /// Also known as `successor` or `next_geq` in the literature.
    ///
    /// ```
    /// use sbits::EliasFano;
    ///
    /// let ef = EliasFano::new(&[10u64, 20, 30, 100, 1000], 2000);
    /// assert_eq!(ef.successor(15), Some(20));
    /// assert_eq!(ef.successor(20), Some(20));
    /// assert_eq!(ef.successor(1001), None);
    /// ```
    pub fn successor(&self, target: u64) -> Option<u64> {
        if self.n == 0 {
            return None;
        }

        let high = (target >> self.l) as usize;
        let lower_mask = if self.l == 0 { 0 } else { (1u64 << self.l) - 1 };
        let target_low = target & lower_mask;

        // select0(high) gives the position in the upper bitvec of the (high+1)-th zero,
        // which is the boundary after bucket `high`. rank1 at that position is bucket_end.
        let (bucket_start, bucket_end) = self.bucket_range_fast(high);

        if bucket_start < bucket_end {
            // Binary search within bucket using only lower bits -- no select1 calls.
            // All elements in this bucket share the same high bits, so comparing lower
            // bits is equivalent to comparing full values.
            let mut lo = bucket_start;
            let mut hi = bucket_end;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.get_lower(mid) < target_low {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo < bucket_end {
                // Reconstruct the value: use get_lower (no select1) since we already have high.
                return Some(((high as u64) << self.l) | self.get_lower(lo));
            }
        }

        // No match in this bucket; successor is the first element in the next non-empty bucket.
        let next_start = bucket_end;
        if next_start < self.n {
            Some(self.get(next_start).unwrap())
        } else {
            None
        }
    }

    /// Return the largest value <= `target`, or `None` if no such value exists.
    ///
    /// Also known as `predecessor` or `prev_leq` in the literature.
    ///
    /// ```
    /// use sbits::EliasFano;
    ///
    /// let ef = EliasFano::new(&[10u64, 20, 30, 100, 1000], 2000);
    /// assert_eq!(ef.predecessor(25), Some(20));
    /// assert_eq!(ef.predecessor(20), Some(20));
    /// assert_eq!(ef.predecessor(9), None);
    /// ```
    pub fn predecessor(&self, target: u64) -> Option<u64> {
        if self.n == 0 {
            return None;
        }

        let high = (target >> self.l) as usize;
        let lower_mask = if self.l == 0 { 0 } else { (1u64 << self.l) - 1 };
        let target_low = target & lower_mask;

        let (bucket_start, bucket_end) = self.bucket_range_fast(high);

        if bucket_start < bucket_end {
            // Binary search within the bucket for last element <= target using only lower bits.
            let mut lo = bucket_start;
            let mut hi = bucket_end;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.get_lower(mid) <= target_low {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo > bucket_start {
                let idx = lo - 1;
                return Some(((high as u64) << self.l) | self.get_lower(idx));
            }
        }

        // No match in this bucket; predecessor is the last element in the previous non-empty bucket.
        if bucket_start > 0 {
            Some(self.get(bucket_start - 1).unwrap())
        } else {
            None
        }
    }

    /// Return the index range [start, end) of elements in the given upper-bits bucket.
    ///
    /// Uses a single `select0` call plus a backward bit scan, avoiding the second `select0`
    /// that the naive two-select approach would need. The backward scan counts consecutive
    /// ones immediately before `end_pos`; that count equals the size of this bucket.
    #[inline]
    fn bucket_range_fast(&self, high: usize) -> (usize, usize) {
        // Position of the (high+1)-th zero = the boundary after bucket `high`.
        let end_pos = match self.upper_bits.select0(high) {
            Some(pos) => pos,
            None => {
                // high is beyond all zeros -- return past-end sentinel.
                return (self.n, self.n);
            }
        };

        // rank1(end_pos) = number of elements with high bits <= high = bucket_end.
        let bucket_end = self.upper_bits.rank1(end_pos);

        // Count the ones immediately preceding end_pos by scanning backward word-by-word.
        // All bits from (previous zero + 1) to (end_pos - 1) are ones; that count is the
        // bucket size. We scan at most one 64-bit word in the common case.
        let bucket_size = self.upper_bits.count_ones_before(end_pos);
        let bucket_start = bucket_end.saturating_sub(bucket_size);

        (bucket_start, bucket_end)
    }

    /// Alias for [`successor`](Self::successor) -- the standard name in IR literature.
    pub fn next_geq(&self, target: u64) -> Option<u64> {
        self.successor(target)
    }

    /// Return the number of encoded values strictly less than `target`.
    ///
    /// # Algorithm
    ///
    /// One `select0(high)` locates the bucket boundary in the upper bitvector.
    /// `bucket_end = select0_pos - high` counts elements with high bits ≤ target's high bits.
    /// A backward walk then subtracts elements whose lower bits ≥ target_low.
    /// Total cost: one select0 + O(bucket_size_match) lower-bit reads,
    /// which is typically 0-1 steps for uniformly spaced data.
    pub fn rank(&self, target: u64) -> usize {
        if self.n == 0 {
            return 0;
        }
        let high = (target >> self.l) as usize;
        let lower_mask = if self.l == 0 { 0 } else { (1u64 << self.l) - 1 };
        let target_low = target & lower_mask;

        // select0(high) gives the position of the (high+1)-th zero in the upper bitvec.
        // That position is right after all elements with high bits == high.
        // rank1 at that position = number of ones before it = number of elements
        // with high bits <= high = bucket_end.
        //
        // high is valid when high < num_upper_zeros = (universe_size >> l) + 1.
        // For target < universe_size, high = target >> l < (universe_size >> l) + 1.
        if high >= self.num_upper_zeros {
            return self.n;
        }
        let h_pos = self.upper_bits.select0_unchecked(high);
        let mut rank = h_pos - high; // = rank1(h_pos) = bucket_end
        let mut pos = h_pos;        // current position in upper bitvec

        // Walk backward: subtract elements in the target bucket whose lower bits >= target_low.
        // Each step: if pos-1 is a 1-bit (still in the same bucket) and lower[rank-1] >= target_low,
        // decrement both rank and pos.
        if self.l == 0 {
            // No lower bits: all elements in bucket count; walk back counting ones before h_pos.
            // target_low is 0 so any element with equal high bits has low bits >= 0, which means
            // all of them are >= target_low only when target_low == 0.
            // Since target_low == 0 always here, we need count of elements with high bits == high.
            // That count is the number of ones immediately before h_pos (the bucket size).
            // But we want elements *strictly less than* target = high<<0 | 0 = high,
            // so elements with high bits < high. That's just bucket_end - bucket_size.
            // bucket_size = count_ones_before(h_pos), but we need to subtract all of them.
            rank -= self.upper_bits.count_ones_before(h_pos);
        } else {
            // Backward walk: subtract elements at the end of this bucket whose lower bits >= target_low.
            while pos > 0 && rank > 0 {
                // pos-1 is always < upper_bits.len() since pos came from select0 (a valid position).
                if !self.upper_bits.get_unchecked(pos - 1) {
                    break; // hit a zero: left the bucket
                }
                if self.get_lower(rank - 1) < target_low {
                    break; // this element has lower bits < target_low: stop
                }
                rank -= 1;
                pos -= 1;
            }
        }

        rank
    }

    /// Return true if `target` is in the encoded sequence.
    ///
    /// Uses `successor` internally: O(log n).
    pub fn contains(&self, target: u64) -> bool {
        self.successor(target) == Some(target)
    }

    /// Return an iterator over all encoded values.
    pub fn iter(&self) -> EliasFanoIter<'_> {
        EliasFanoIter { ef: self, idx: 0 }
    }

    /// Heap memory usage in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.upper_bits.heap_bytes() + self.lower_bits.len() * 8
    }

    /// Serialize this Elias–Fano structure to a stable binary encoding (little-endian).
    ///
    /// Format (versioned):
    /// - magic: 8 bytes (`SBITEF02`)
    /// - universe_size: u64
    /// - l: u32
    /// - n: u64
    /// - lower_len: u64, then `lower_len` u64 words
    /// - upper_bits: byte_len u64, then `byte_len` bytes (BitVector::to_bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITEF02");
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
        use crate::error::ByteReader;
        let mut r = ByteReader::new(bytes);
        r.read_magic(b"SBITEF02", "EliasFano")?;
        let universe_size = r.read_u64()?;
        let l = r.read_u32()? as usize;
        if l > 63 {
            return Err(Error::InvalidEncoding(format!(
                "EliasFano l={l} exceeds maximum (63) for u64 values"
            )));
        }
        let n = r.read_u64()? as usize;
        let lower_len = r.read_u64()? as usize;
        let lower_bits = r.read_u64_vec(lower_len)?;
        let upper_len = r.read_u64()? as usize;
        let upper_bytes = r.take(upper_len)?;
        let upper_bits = BitVector::from_bytes(upper_bytes)?;
        r.expect_eof("EliasFano")?;

        let num_upper_zeros = (universe_size >> l) as usize + 1;
        Ok(Self {
            universe_size,
            upper_bits,
            lower_bits,
            l,
            n,
            num_upper_zeros,
        })
    }
}

/// Iterator over values in an [`EliasFano`] structure.
pub struct EliasFanoIter<'a> {
    ef: &'a EliasFano,
    idx: usize,
}

impl Iterator for EliasFanoIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.idx >= self.ef.n {
            return None;
        }
        let val = self.ef.get(self.idx).unwrap();
        self.idx += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ef.n - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EliasFanoIter<'_> {}

impl<'a> IntoIterator for &'a EliasFano {
    type Item = u64;
    type IntoIter = EliasFanoIter<'a>;

    fn into_iter(self) -> EliasFanoIter<'a> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_elias_fano_basic() {
        let values = vec![10u64, 20, 30, 100, 1000];
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
        let values = vec![0u64, 1, 2, 3];
        let ef = EliasFano::new(&values, 4);
        assert_eq!(ef.len(), 4);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef.get(i).unwrap(), v, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_elias_fano_serialization_roundtrip() {
        let values = vec![10u64, 20, 30, 100, 1000];
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
        let values = vec![0u64, 1, 2, 3];
        let ef = EliasFano::new(&values, 4);
        let bytes = ef.to_bytes();
        let ef2 = EliasFano::from_bytes(&bytes).unwrap();
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(ef2.get(i).unwrap(), v);
        }
    }

    #[test]
    fn test_elias_fano_rejects_bad_l() {
        let ef = EliasFano::new(&[10u64], 100);
        let mut bytes = ef.to_bytes();
        // Corrupt the `l` field (offset 16..20) to 64.
        bytes[16] = 64;
        bytes[17] = 0;
        bytes[18] = 0;
        bytes[19] = 0;
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

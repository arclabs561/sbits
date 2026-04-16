//! Cache-friendly succinct bit vector with rank/select support.
//!
//! Implements the Rank9 indexing scheme with an interleaved (blocked) layout
//! for superior cache locality.
//!
//! # Layout
//!
//! Each 512-bit block is stored as 10 x 64-bit words:
//! - Word 0: Absolute rank (number of 1s before this block)
//! - Word 1: Relative ranks (7 x 9-bit cumulative counts within the block)
//! - Word 2-9: Raw data (512 bits)
//!
//! This ensures that once the block header is in cache, all data needed for
//! rank/select within that block is also available.

use alloc::vec;
use alloc::vec::Vec;
use alloc::{format, string::ToString};

/// A cache-oblivious succinct bit vector.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitVector {
    /// Interleaved data: [abs_rank, rel_ranks, data0, ..., data7, ...]
    storage: Vec<u64>,
    /// Coarse index for select1: stores block index for every 512th one-bit
    select1_index: Vec<u32>,
    /// Coarse index for select0: stores block index for every 512th zero-bit
    select0_index: Vec<u32>,
    len: usize,
}

impl core::fmt::Debug for BitVector {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BitVector")
            .field("len", &self.len)
            .field("ones", &self.rank1(self.len))
            .finish()
    }
}

impl BitVector {
    /// Create a new BitVector from a sequence of bits.
    ///
    /// ```
    /// use sbits::BitVector;
    ///
    /// let bits = vec![0b1011u64]; // bits 0, 1, 3 are set
    /// let bv = BitVector::new(&bits, 64);
    /// assert!(bv.get(0));
    /// assert!(!bv.get(2));
    /// assert_eq!(bv.rank1(4), 3); // three 1-bits in [0, 4)
    /// assert_eq!(bv.select1(2), Some(3)); // third 1-bit is at position 3
    /// ```
    pub fn new(bits: &[u64], len: usize) -> Self {
        let num_blocks = len.div_ceil(512);
        let mut storage = vec![0u64; num_blocks * 10 + 10]; // +10 for sentinel
        let mut select1_index = Vec::new();
        let mut select0_index = Vec::new();

        let mut total_rank = 0u64;
        let mut next_select1_threshold = 0u64;
        let mut next_select0_threshold = 0u64;

        for i in 0..num_blocks {
            let base = i * 10;
            storage[base] = total_rank;
            let total_zeros = (i as u64 * 512) - total_rank;

            while total_rank >= next_select1_threshold {
                select1_index.push(i as u32);
                next_select1_threshold += 512;
            }
            while total_zeros >= next_select0_threshold {
                select0_index.push(i as u32);
                next_select0_threshold += 512;
            }

            let mut relative_ranks = 0u64;
            let mut current_rel = 0u64;

            for j in 0..8 {
                let data_idx = i * 8 + j;
                let word = if data_idx < bits.len() {
                    bits[data_idx]
                } else {
                    0
                };
                storage[base + 2 + j] = word;

                if j > 0 {
                    relative_ranks |= current_rel << (9 * (j - 1));
                }
                current_rel += word.count_ones() as u64;
            }
            storage[base + 1] = relative_ranks;
            total_rank += current_rel;
        }

        // Sentinel
        let last_base = num_blocks * 10;
        storage[last_base] = total_rank;
        let total_zeros = (num_blocks as u64 * 512) - total_rank;
        while total_rank >= next_select1_threshold {
            select1_index.push(num_blocks as u32);
            next_select1_threshold += 512;
        }
        while total_zeros >= next_select0_threshold {
            select0_index.push(num_blocks as u32);
            next_select0_threshold += 512;
        }

        Self {
            storage,
            select1_index,
            select0_index,
            len,
        }
    }

    /// Create a `BitVector` from an iterator of set-bit positions.
    ///
    /// ```
    /// use sbits::BitVector;
    ///
    /// let bv = BitVector::from_ones([0, 1, 3, 7].into_iter(), 16);
    /// assert_eq!(bv.rank1(4), 3);
    /// assert_eq!(bv.select1(3), Some(7));
    /// ```
    pub fn from_ones(positions: impl Iterator<Item = usize>, len: usize) -> Self {
        let num_words = len.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for pos in positions {
            if pos < len {
                bits[pos / 64] |= 1u64 << (pos % 64);
            }
        }
        Self::new(&bits, len)
    }

    /// Reconstruct a `BitVector` from its internal parts.
    ///
    /// This is primarily intended for serialization round-trips.
    pub fn from_parts(
        storage: Vec<u64>,
        select1_index: Vec<u32>,
        select0_index: Vec<u32>,
        len: usize,
    ) -> crate::error::Result<Self> {
        // Structural validation to prevent OOB panics in rank/select.
        if storage.len() < 10 {
            return Err(crate::error::Error::InvalidEncoding(
                "bitvec storage too small (need >= 10 words)".to_string(),
            ));
        }
        if !storage.len().is_multiple_of(10) {
            return Err(crate::error::Error::InvalidEncoding(
                "bitvec storage len must be multiple of 10".to_string(),
            ));
        }
        // num_blocks = (storage.len() / 10) - 1  (last block is sentinel).
        let num_blocks = storage.len() / 10 - 1;
        let max_bits = num_blocks * 512;
        if len > max_bits {
            return Err(crate::error::Error::InvalidEncoding(format!(
                "bitvec len ({len}) exceeds capacity of {num_blocks} blocks ({max_bits} bits)"
            )));
        }

        Ok(Self {
            storage,
            select1_index,
            select0_index,
            len,
        })
    }

    /// Serialize this bitvector to a stable binary encoding (little-endian).
    ///
    /// Format (versioned):
    /// - magic: 8 bytes (`SBITBV01`)
    /// - len: u64
    /// - storage_len: u64, then `storage_len` u64 words
    /// - select1_len: u64, then `select1_len` u32 words
    /// - select0_len: u64, then `select0_len` u32 words
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITBV01");

        out.extend_from_slice(&(self.len as u64).to_le_bytes());

        out.extend_from_slice(&(self.storage.len() as u64).to_le_bytes());
        for &w in &self.storage {
            out.extend_from_slice(&w.to_le_bytes());
        }

        out.extend_from_slice(&(self.select1_index.len() as u64).to_le_bytes());
        for &w in &self.select1_index {
            out.extend_from_slice(&w.to_le_bytes());
        }

        out.extend_from_slice(&(self.select0_index.len() as u64).to_le_bytes());
        for &w in &self.select0_index {
            out.extend_from_slice(&w.to_le_bytes());
        }

        out
    }

    /// Deserialize a `BitVector` from `to_bytes()` output.
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        use crate::error::ByteReader;
        let mut r = ByteReader::new(bytes);
        r.read_magic(b"SBITBV01", "BitVector")?;
        let len = r.read_u64()? as usize;
        let storage_len = r.read_u64()? as usize;
        let storage = r.read_u64_vec(storage_len)?;
        let select1_len = r.read_u64()? as usize;
        let select1_index = r.read_u32_vec(select1_len)?;
        let select0_len = r.read_u64()? as usize;
        let select0_index = r.read_u32_vec(select0_len)?;
        r.expect_eof("BitVector")?;
        Self::from_parts(storage, select1_index, select0_index, len)
    }

    /// Return the total number of bits in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true if the bit-vector has length 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Heap memory usage in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.storage.len() * 8 + self.select1_index.len() * 4 + self.select0_index.len() * 4
    }

    /// Return true if the bit at index `i` is set.
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        let block_idx = i / 512;
        let word_in_block = (i % 512) / 64;
        let bit_in_word = i % 64;
        let word = self.storage[block_idx * 10 + 2 + word_in_block];
        (word & (1u64 << bit_in_word)) != 0
    }

    /// Fast unchecked bit access for known-valid positions.
    ///
    /// Like `get()` but skips the bounds check. The caller must ensure `i < self.len`.
    /// Used by EliasFano's rank backward walk to avoid repeated bounds checks.
    #[inline(always)]
    pub(crate) fn get_unchecked(&self, i: usize) -> bool {
        let block_idx = i / 512;
        let word_in_block = (i % 512) / 64;
        let bit_in_word = i % 64;
        let word = self.storage[block_idx * 10 + 2 + word_in_block];
        (word >> bit_in_word) & 1 != 0
    }

    /// Return the number of set bits in the range [0, i).
    pub fn rank1(&self, i: usize) -> usize {
        if i == 0 {
            return 0;
        }
        let i = i.min(self.len);
        let block_idx = i / 512;
        let sub_block_idx = (i % 512) / 64;
        let bit_offset = i % 64;

        let base = block_idx * 10;
        let mut rank = self.storage[base] as usize;

        if sub_block_idx > 0 {
            let relative_ranks = self.storage[base + 1];
            rank += ((relative_ranks >> (9 * (sub_block_idx - 1))) & 0x1FF) as usize;
        }

        let word = self.storage[base + 2 + sub_block_idx];
        // bit_offset = i % 64, so it is always in [0, 63].
        let mask = (1u64 << bit_offset).wrapping_sub(1);
        rank += (word & mask).count_ones() as usize;

        rank
    }

    /// Return the number of unset bits in the range [0, i).
    pub fn rank0(&self, i: usize) -> usize {
        i - self.rank1(i)
    }

    /// Return the position of the $k$-th set bit (0-indexed).
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.rank1(self.len) {
            return None;
        }

        let target = k + 1;
        let select_idx = k / 512;
        let mut block_low = self.select1_index[select_idx] as usize;
        let mut block_high = if select_idx + 1 < self.select1_index.len() {
            self.select1_index[select_idx + 1] as usize + 1
        } else {
            self.storage.len() / 10
        };

        while block_low < block_high {
            let mid = block_low + (block_high - block_low) / 2;
            if self.storage[mid * 10] < target as u64 {
                block_low = mid + 1;
            } else {
                block_high = mid;
            }
        }
        let block_idx = block_low - 1;
        let mut remaining_k = target - (self.storage[block_idx * 10] as usize);

        let relative_ranks = self.storage[block_idx * 10 + 1];
        let mut sub_block_idx = 0;
        for j in 1..8 {
            let rel_rank = ((relative_ranks >> (9 * (j - 1))) & 0x1FF) as usize;
            if rel_rank < remaining_k {
                sub_block_idx = j;
            } else {
                break;
            }
        }

        if sub_block_idx > 0 {
            let rel_rank = ((relative_ranks >> (9 * (sub_block_idx - 1))) & 0x1FF) as usize;
            remaining_k -= rel_rank;
        }

        let word = self.storage[block_idx * 10 + 2 + sub_block_idx];
        let pos_in_word = self.select_in_word(word, remaining_k - 1);
        Some(block_idx * 512 + sub_block_idx * 64 + pos_in_word)
    }

    /// Return the position of the $k$-th unset bit (0-indexed).
    pub fn select0(&self, k: usize) -> Option<usize> {
        if k >= self.rank0(self.len) {
            return None;
        }
        Some(self.select0_inner(k))
    }

    /// `select0` without the bounds check; caller must ensure `k < rank0(len)`.
    ///
    /// Saves the `rank1(len)` call that `select0` uses for bounds checking.
    /// Used by EliasFano where `k = target >> l` is always within the valid range.
    #[inline]
    pub(crate) fn select0_unchecked(&self, k: usize) -> usize {
        self.select0_inner(k)
    }

    #[inline]
    fn select0_inner(&self, k: usize) -> usize {
        let target = k + 1;
        let select_idx = k / 512;
        let mut block_low = self.select0_index[select_idx] as usize;
        let mut block_high = if select_idx + 1 < self.select0_index.len() {
            self.select0_index[select_idx + 1] as usize + 1
        } else {
            self.storage.len() / 10
        };

        while block_low < block_high {
            let mid = block_low + (block_high - block_low) / 2;
            let rank0_at_mid = (mid * 512) - (self.storage[mid * 10] as usize);
            if rank0_at_mid < target {
                block_low = mid + 1;
            } else {
                block_high = mid;
            }
        }
        let block_idx = block_low - 1;
        let mut remaining_k =
            target - ((block_idx * 512) - (self.storage[block_idx * 10] as usize));

        let relative_ranks1 = self.storage[block_idx * 10 + 1];
        let mut sub_block_idx = 0;
        for j in 1..8 {
            let rel_rank1 = ((relative_ranks1 >> (9 * (j - 1))) & 0x1FF) as usize;
            let rel_rank0 = (j * 64) - rel_rank1;
            if rel_rank0 < remaining_k {
                sub_block_idx = j;
            } else {
                break;
            }
        }

        if sub_block_idx > 0 {
            let rel_rank1 = ((relative_ranks1 >> (9 * (sub_block_idx - 1))) & 0x1FF) as usize;
            remaining_k -= (sub_block_idx * 64) - rel_rank1;
        }

        let word = !self.storage[block_idx * 10 + 2 + sub_block_idx];
        let pos_in_word = self.select_in_word(word, remaining_k - 1);
        block_idx * 512 + sub_block_idx * 64 + pos_in_word
    }

    /// Count the number of consecutive set bits immediately before position `pos`.
    ///
    /// Scans backward from `pos - 1`, stopping at the first unset bit or the start.
    /// Used by EliasFano's `bucket_range_fast` to find a bucket's start element count
    /// without a second `select0` call.
    pub(crate) fn count_ones_before(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.len);
        let mut count = 0usize;

        // Raw word index in the data portion (ignoring block headers).
        // Block-interleaved: each block is 10 words [abs_rank, rel_ranks, data0..data7].
        // Raw data word r maps to storage[r/8 * 10 + 2 + r%8].
        #[inline(always)]
        fn raw_to_storage(r: usize) -> usize {
            (r / 8) * 10 + 2 + (r % 8)
        }

        let end_bit = pos - 1; // highest bit index to examine
        let mut raw_word = end_bit / 64;
        let bit_in_word = end_bit % 64; // bit position within first word (0 = LSB)

        // First (possibly partial) word: examine bits [0..=bit_in_word].
        // We want consecutive 1s from bit_in_word downward.
        // Strategy: reverse the relevant bits and count trailing zeros of the complement.
        let w = self.storage[raw_to_storage(raw_word)];
        // Shift so that bit_in_word becomes bit 63; higher bits are shifted out.
        let shifted = w << (63 - bit_in_word);
        // leading_zeros(!shifted) = number of consecutive 1s from bit_in_word downward.
        let ones = (!shifted).leading_zeros() as usize;
        count += ones;
        if ones <= bit_in_word {
            // Hit a zero -- done.
            return count;
        }
        // Entire relevant portion of this word was ones (ones == bit_in_word + 1).
        if raw_word == 0 {
            return count;
        }
        raw_word -= 1;

        // Full words: scan backward while all 64 bits are set.
        loop {
            let w = self.storage[raw_to_storage(raw_word)];
            let ones = (!w).leading_zeros() as usize;
            count += ones;
            if ones < 64 {
                break;
            }
            if raw_word == 0 {
                break;
            }
            raw_word -= 1;
        }

        count
    }

    fn select_in_word(&self, word: u64, k: usize) -> usize {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            unsafe {
                let mask = 1u64 << k;
                let res = core::arch::x86_64::_pdep_u64(mask, word);
                return res.trailing_zeros() as usize;
            }
        }

        // Broadword select algorithm (Vigna, "Broadword Implementation of Rank/Select Queries").
        // O(1) arithmetic operations, no branches dependent on data.
        // Reference: https://vigna.di.unimi.it/ftp/papers/Broadword.pdf
        const L8: u64 = 0x0101_0101_0101_0101u64; // 1 in every byte
        const H8: u64 = 0x8080_8080_8080_8080u64; // high bit of every byte

        // Step 1: compute byte-level cumulative popcount using broadword.
        // s[i] = number of set bits in bytes 0..=i of `word`.
        let mut s = word.wrapping_sub((word >> 1) & 0x5555_5555_5555_5555u64);
        s = (s & 0x3333_3333_3333_3333u64) + ((s >> 2) & 0x3333_3333_3333_3333u64);
        s = s.wrapping_add(s >> 4) & 0x0F0F_0F0F_0F0F_0F0Fu64;
        // Now s has per-byte popcounts. Compute prefix sums across bytes via multiply.
        // After multiply by L8: byte i of result = sum of bytes 0..=i of s.
        let byte_sums = s.wrapping_mul(L8);

        // Find which byte contains the k-th bit using ULEQ comparison.
        // k_step = k replicated in each byte (k < 64, so this fits in 7 bits).
        let k_step8 = (k as u64).wrapping_mul(L8);
        // leq_places[i] = 0x80 if byte_sums[i] <= k_step8[i], else 0.
        // We want the first byte where cumulative count > k, i.e., where NOT leq.
        let geq_step8 = ((k_step8 | H8).wrapping_sub(byte_sums & !H8)) & H8;
        // Count bytes where cumulative count <= k: that's the byte index to search within.
        let byte_idx = (geq_step8.count_ones() as usize) * 8;

        // Within the identified byte, strip `remaining` set bits.
        let remaining = k - (byte_sums >> byte_idx).wrapping_sub(1) as u8 as usize;
        let byte_word = (word >> byte_idx) as u8 as u64;
        let mut w = byte_word;
        for _ in 0..remaining {
            w &= w.wrapping_sub(1);
        }
        byte_idx + w.trailing_zeros() as usize
    }

    /// Return an iterator over the positions of all set bits.
    pub fn ones(&self) -> OnesIter<'_> {
        OnesIter {
            bv: self,
            remaining: self.rank1(self.len),
            block: 0,
            word_in_block: 0,
            current_word: self.first_data_word(0),
            base_pos: 0,
        }
    }

    /// Return an iterator over the positions of all unset bits (up to `len`).
    pub fn zeros(&self) -> ZerosIter<'_> {
        ZerosIter {
            bv: self,
            remaining: self.rank0(self.len),
            block: 0,
            word_in_block: 0,
            current_word: self.first_data_word_inverted(0),
            base_pos: 0,
        }
    }

    fn first_data_word(&self, block: usize) -> u64 {
        if block * 10 + 2 < self.storage.len() {
            self.storage[block * 10 + 2]
        } else {
            0
        }
    }

    fn first_data_word_inverted(&self, block: usize) -> u64 {
        if block * 10 + 2 < self.storage.len() {
            !self.storage[block * 10 + 2]
        } else {
            0
        }
    }
}

/// Iterator over set-bit positions in a [`BitVector`].
pub struct OnesIter<'a> {
    bv: &'a BitVector,
    remaining: usize,
    block: usize,
    word_in_block: usize,
    current_word: u64,
    base_pos: usize,
}

impl Iterator for OnesIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word.wrapping_sub(1);
                let pos = self.base_pos + bit;
                if pos < self.bv.len {
                    self.remaining -= 1;
                    return Some(pos);
                }
                return None;
            }
            self.word_in_block += 1;
            if self.word_in_block >= 8 {
                self.block += 1;
                self.word_in_block = 0;
            }
            let idx = self.block * 10 + 2 + self.word_in_block;
            if idx >= self.bv.storage.len() {
                return None;
            }
            self.base_pos = self.block * 512 + self.word_in_block * 64;
            if self.base_pos >= self.bv.len {
                return None;
            }
            self.current_word = self.bv.storage[idx];
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for OnesIter<'_> {}

/// Iterator over unset-bit positions in a [`BitVector`].
pub struct ZerosIter<'a> {
    bv: &'a BitVector,
    remaining: usize,
    block: usize,
    word_in_block: usize,
    current_word: u64,
    base_pos: usize,
}

impl Iterator for ZerosIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word.wrapping_sub(1);
                let pos = self.base_pos + bit;
                if pos < self.bv.len {
                    self.remaining -= 1;
                    return Some(pos);
                }
                return None;
            }
            self.word_in_block += 1;
            if self.word_in_block >= 8 {
                self.block += 1;
                self.word_in_block = 0;
            }
            let idx = self.block * 10 + 2 + self.word_in_block;
            if idx >= self.bv.storage.len() {
                return None;
            }
            self.base_pos = self.block * 512 + self.word_in_block * 64;
            if self.base_pos >= self.bv.len {
                return None;
            }
            self.current_word = !self.bv.storage[idx];
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for ZerosIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_bitvector_rank_basic() {
        let data = vec![0b1011, 0b1101];
        let bv = BitVector::new(&data, 128);
        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(1), 1);
        assert_eq!(bv.rank1(4), 3);
        assert!(bv.get(0));
        assert!(!bv.get(2));
    }

    #[test]
    fn test_bitvector_select_basic() {
        let data = vec![0b1011];
        let bv = BitVector::new(&data, 64);
        assert_eq!(bv.select1(0), Some(0));
        assert_eq!(bv.select1(1), Some(1));
        assert_eq!(bv.select1(2), Some(3));
        assert_eq!(bv.select1(3), None);

        assert_eq!(bv.select0(0), Some(2));
        assert_eq!(bv.select0(1), Some(4));
    }

    #[test]
    fn test_bitvector_serialization_roundtrip() {
        let data = vec![0b1011, 0b1101];
        let bv = BitVector::new(&data, 128);
        let bytes = bv.to_bytes();
        let bv2 = BitVector::from_bytes(&bytes).unwrap();
        assert_eq!(bv2.len(), 128);
        assert_eq!(bv2.rank1(4), 3);
        assert!(bv2.get(0));
        assert!(!bv2.get(2));
    }

    #[test]
    fn test_bitvector_from_parts_rejects_bad_len() {
        // 10 words = 1 block, max 512 bits. len=513 should fail.
        let storage = vec![0u64; 20]; // 2 blocks (1 real + sentinel)
        assert!(BitVector::from_parts(storage.clone(), vec![], vec![], 512).is_ok());
        assert!(BitVector::from_parts(storage, vec![], vec![], 513).is_err());
    }

    #[test]
    fn test_bitvector_from_bytes_rejects_allocation_bomb() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SBITBV01");
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(u64::MAX).to_le_bytes());
        assert!(BitVector::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_bitvector_empty() {
        let bv = BitVector::new(&[], 0);
        assert!(bv.is_empty());
        assert_eq!(bv.len(), 0);
        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank0(0), 0);
        assert_eq!(bv.select1(0), None);
        assert_eq!(bv.select0(0), None);
        assert!(!bv.get(0));
        assert_eq!(bv.ones().count(), 0);
        assert_eq!(bv.zeros().count(), 0);
    }

    #[test]
    fn test_bitvector_get_oob_returns_false() {
        let bv = BitVector::new(&[0xFFFF], 16);
        assert!(bv.get(0));
        assert!(bv.get(15));
        assert!(!bv.get(16));
        assert!(!bv.get(1000));
    }

    #[test]
    fn test_bitvector_cross_block_boundary() {
        // 9 words of all-ones = 576 bits, crossing the 512-bit block boundary.
        let data = vec![u64::MAX; 9];
        let bv = BitVector::new(&data, 576);
        assert_eq!(bv.rank1(512), 512);
        assert_eq!(bv.rank1(576), 576);
        for k in 0..576 {
            assert_eq!(bv.select1(k), Some(k), "select1({k}) failed");
        }
        assert_eq!(bv.select1(576), None);
    }

    #[test]
    fn test_bitvector_serialization_verifies_select() {
        let data = vec![0b1011, 0b1101];
        let bv = BitVector::new(&data, 128);
        let bytes = bv.to_bytes();
        let bv2 = BitVector::from_bytes(&bytes).unwrap();
        // Verify select works after deserialization
        assert_eq!(bv2.select1(0), bv.select1(0));
        assert_eq!(bv2.select1(2), bv.select1(2));
        assert_eq!(bv2.select0(0), bv.select0(0));
        assert_eq!(bv2.select0(2), bv.select0(2));
    }

    #[test]
    fn test_bitvector_ones_iter() {
        let bv = BitVector::new(&[0b1011], 64);
        let ones: Vec<usize> = bv.ones().collect();
        assert_eq!(ones, vec![0, 1, 3]);
    }

    #[test]
    fn test_bitvector_zeros_iter() {
        let bv = BitVector::new(&[0b1011], 4);
        let zeros: Vec<usize> = bv.zeros().collect();
        assert_eq!(zeros, vec![2]);
    }
}

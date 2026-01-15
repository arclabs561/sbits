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

/// A cache-oblivious succinct bit vector.
pub struct BitVector {
    /// Interleaved data: [abs_rank, rel_ranks, data0, ..., data7, ...]
    storage: Vec<u64>,
    /// Coarse index for select1: stores block index for every 512th one-bit
    select1_index: Vec<u32>,
    /// Coarse index for select0: stores block index for every 512th zero-bit
    select0_index: Vec<u32>,
    len: usize,
}

impl std::fmt::Debug for BitVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitVector")
            .field("len", &self.len)
            .field("ones", &self.rank1(self.len))
            .finish()
    }
}

impl BitVector {
    /// Create a new BitVector from a sequence of bits.
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

    /// Return the total number of bits in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true if the bit-vector has length 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Approximate heap memory usage in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.storage.capacity() * 8
            + self.select1_index.capacity() * 4
            + self.select0_index.capacity() * 4
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
        let mask = if bit_offset == 64 {
            !0u64
        } else {
            (1u64 << bit_offset).wrapping_sub(1)
        };
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
            let rel_rank0 = (sub_block_idx * 64) - rel_rank1;
            remaining_k -= rel_rank0;
        }

        let word = !self.storage[block_idx * 10 + 2 + sub_block_idx];
        let pos_in_word = self.select_in_word(word, remaining_k - 1);
        Some(block_idx * 512 + sub_block_idx * 64 + pos_in_word)
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

        let mut count = 0;
        for i in 0..64 {
            if (word & (1 << i)) != 0 {
                if count == k {
                    return i;
                }
                count += 1;
            }
        }
        63
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

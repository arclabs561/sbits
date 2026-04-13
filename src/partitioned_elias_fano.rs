//! Partitioned Elias–Fano for clustered monotone sequences.
//!
//! Intuition: plain Elias–Fano chooses a single `L = floor(log2(U/n))` based on the *global*
//! universe size. If a sequence is locally clustered (common in posting lists and ANN neighbor
//! lists), using a single global `L` can waste bits. Partitioned Elias–Fano splits the sequence
//! into blocks and encodes each block with its own local universe, improving compression while
//! preserving fast random access via per-block decoding.
//!
//! This implementation is intentionally simple:
//! - blocks are encoded as independent `EliasFano` structures over values shifted by the block base
//! - access is `O(1)` block indexing + `O(1)` `EliasFano::get`
//! - serialization is stable and versioned
//!
//! It assumes the input values are sorted (monotone) and all `< universe_size`.

use crate::elias_fano::EliasFano;
use crate::error::{Error, Result};

/// Partitioned Elias–Fano encoding.
#[derive(Debug, Clone)]
pub struct PartitionedEliasFano {
    universe_size: u64,
    block_size: usize,
    n: usize,
    bases: Vec<u64>,
    blocks: Vec<EliasFano>,
}

impl PartitionedEliasFano {
    /// Build a partitioned Elias–Fano structure from a sorted sequence.
    ///
    /// `block_size` is the maximum number of items per block (must be >= 1; values 64–256 are
    /// typical engineering choices).
    #[must_use]
    pub fn new(values: &[u64], universe_size: u64, block_size: usize) -> Self {
        let n = values.len();
        let block_size = block_size.max(1);
        if n == 0 {
            return Self {
                universe_size,
                block_size,
                n: 0,
                bases: Vec::new(),
                blocks: Vec::new(),
            };
        }

        let mut bases = Vec::new();
        let mut blocks = Vec::new();
        let mut i = 0usize;
        while i < n {
            let j = (i + block_size).min(n);
            let base = values[i];
            let last = values[j - 1];
            let local_u = (last - base).saturating_add(1);
            let rel: Vec<u64> = values[i..j].iter().map(|&v| v - base).collect();
            bases.push(base);
            blocks.push(EliasFano::new(&rel, local_u));
            i = j;
        }

        Self {
            universe_size,
            block_size,
            n,
            bases,
            blocks,
        }
    }

    /// Return the universe size used to build this structure.
    #[must_use]
    pub fn universe_size(&self) -> u64 {
        self.universe_size
    }

    /// Return the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return true if the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Maximum number of values per block.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Return the value at index `i`.
    pub fn get(&self, i: usize) -> Result<u64> {
        if i >= self.n {
            return Err(Error::IndexOutOfBounds(i));
        }
        let b = i / self.block_size;
        let off = i % self.block_size;
        let base = *self
            .bases
            .get(b)
            .ok_or(Error::InvalidEncoding("missing block base".to_string()))?;
        let block = self
            .blocks
            .get(b)
            .ok_or(Error::InvalidEncoding("missing block".to_string()))?;
        let rel = block.get(off)?;
        Ok(base + rel)
    }

    /// Return an iterator over all encoded values.
    pub fn iter(&self) -> PefIter<'_> {
        PefIter {
            pef: self,
            block_idx: 0,
            elem_in_block: 0,
            remaining: self.n,
        }
    }

    /// Heap memory usage in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.bases.len() * 8 + self.blocks.iter().map(|b| b.heap_bytes()).sum::<usize>()
    }

    /// Serialize this partitioned structure to a stable binary encoding (little-endian).
    ///
    /// Format (versioned):
    /// - magic: 8 bytes (`SBITPEF2`)
    /// - universe_size: u64
    /// - block_size: u32
    /// - n: u64
    /// - num_blocks: u64
    /// - bases: `num_blocks` u64
    /// - blocks: for each block: len_bytes u64, then `len_bytes` bytes (EliasFano::to_bytes)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITPEF2");
        out.extend_from_slice(&self.universe_size.to_le_bytes());
        out.extend_from_slice(&(self.block_size as u32).to_le_bytes());
        out.extend_from_slice(&(self.n as u64).to_le_bytes());
        out.extend_from_slice(&(self.blocks.len() as u64).to_le_bytes());

        for &b in &self.bases {
            out.extend_from_slice(&b.to_le_bytes());
        }
        for blk in &self.blocks {
            let bytes = blk.to_bytes();
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    /// Deserialize a partitioned Elias–Fano structure from `to_bytes()` output.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use crate::error::ByteReader;
        let mut r = ByteReader::new(bytes);
        r.read_magic(b"SBITPEF2", "PartitionedEliasFano")?;
        let universe_size = r.read_u64()?;
        let block_size = r.read_u32()? as usize;
        let n = r.read_u64()? as usize;
        let num_blocks = r.read_u64()? as usize;
        let bases = r.read_u64_vec(num_blocks)?;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let len_bytes = r.read_u64()? as usize;
            let blk_bytes = r.take(len_bytes)?;
            let ef = EliasFano::from_bytes(blk_bytes)?;
            blocks.push(ef);
        }
        r.expect_eof("PartitionedEliasFano")?;

        if block_size == 0 {
            return Err(Error::InvalidEncoding(
                "block_size must be >= 1".to_string(),
            ));
        }
        let actual_n: usize = blocks.iter().map(|b| b.len()).sum();
        if actual_n != n {
            return Err(Error::InvalidEncoding(format!(
                "PEF n ({n}) does not match sum of block lengths ({actual_n})"
            )));
        }

        Ok(Self {
            universe_size,
            block_size,
            n,
            bases,
            blocks,
        })
    }
}

/// Iterator over values in a [`PartitionedEliasFano`] structure.
///
/// Iterates block-by-block to avoid per-element division/modulo overhead.
pub struct PefIter<'a> {
    pef: &'a PartitionedEliasFano,
    block_idx: usize,
    elem_in_block: usize,
    remaining: usize,
}

impl Iterator for PefIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.remaining == 0 {
            return None;
        }
        // Advance to next block if current is exhausted.
        while self.block_idx < self.pef.blocks.len()
            && self.elem_in_block >= self.pef.blocks[self.block_idx].len()
        {
            self.block_idx += 1;
            self.elem_in_block = 0;
        }
        if self.block_idx >= self.pef.blocks.len() {
            return None;
        }
        let base = self.pef.bases[self.block_idx];
        let rel = self.pef.blocks[self.block_idx]
            .get(self.elem_in_block)
            .unwrap();
        self.elem_in_block += 1;
        self.remaining -= 1;
        Some(base + rel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PefIter<'_> {}

impl<'a> IntoIterator for &'a PartitionedEliasFano {
    type Item = u64;
    type IntoIter = PefIter<'a>;

    fn into_iter(self) -> PefIter<'a> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitioned_roundtrip_basic() {
        let values = vec![10u64, 20, 30, 31, 32, 100, 1000];
        let pef = PartitionedEliasFano::new(&values, 2000, 3);
        assert_eq!(pef.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.get(i).unwrap(), v);
        }

        let bytes = pef.to_bytes();
        let pef2 = PartitionedEliasFano::from_bytes(&bytes).unwrap();
        assert_eq!(pef2.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef2.get(i).unwrap(), v);
        }
    }

    #[test]
    fn partitioned_single_element() {
        let pef = PartitionedEliasFano::new(&[42u64], 100, 64);
        assert_eq!(pef.len(), 1);
        assert_eq!(pef.get(0).unwrap(), 42);
        let bytes = pef.to_bytes();
        let pef2 = PartitionedEliasFano::from_bytes(&bytes).unwrap();
        assert_eq!(pef2.get(0).unwrap(), 42);
    }

    #[test]
    fn partitioned_empty() {
        let pef = PartitionedEliasFano::new(&[], 100u64, 64);
        assert!(pef.is_empty());
        assert!(pef.get(0).is_err());
    }

    #[test]
    fn partitioned_block_boundary() {
        // block_size=3, 6 elements = exactly 2 full blocks.
        let values = vec![0u64, 1, 2, 10, 11, 12];
        let pef = PartitionedEliasFano::new(&values, 20, 3);
        assert_eq!(pef.num_blocks(), 2);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.get(i).unwrap(), v);
        }
    }

    #[test]
    fn partitioned_block_size_larger_than_n() {
        let values = vec![5u64, 10, 15];
        let pef = PartitionedEliasFano::new(&values, 20, 100);
        assert_eq!(pef.num_blocks(), 1);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(pef.get(i).unwrap(), v);
        }
    }

    #[test]
    fn partitioned_rejects_corrupted_n() {
        let values = vec![10u64, 20, 30];
        let pef = PartitionedEliasFano::new(&values, 100, 2);
        let mut bytes = pef.to_bytes();
        // Corrupt the `n` field (offset 20..28) to a wrong value.
        let bad_n: u64 = 999;
        bytes[20..28].copy_from_slice(&bad_n.to_le_bytes());
        assert!(PartitionedEliasFano::from_bytes(&bytes).is_err());
    }
}

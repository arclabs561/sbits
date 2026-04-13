//! Wavelet Tree for arbitrary alphabets.
//!
//! Generalizes rank and select operations from bit vectors to
//! sequences over larger alphabets $\Sigma$.
//!
//! # Theory
//!
//! A Wavelet Tree for a string $S$ of length $n$ over alphabet $\Sigma$:
//! - Root node partitions $\Sigma$ into two halves $\Sigma_L, \Sigma_R$.
//! - A bit vector at the root marks if $S\[i\] \in \Sigma_R$.
//! - Left child is Wavelet Tree for $S$ restricted to $\Sigma_L$.
//! - Right child is Wavelet Tree for $S$ restricted to $\Sigma_R$.
//!
//! Total space: $n \log |\Sigma| + o(n \log |\Sigma|)$ bits.
//! Queries `access`, `rank`, `select` take $O(\log |\Sigma|)$ time.

use crate::bitvec::BitVector;
use crate::error::{ByteReader, Error, Result};

/// Wavelet Tree node.
#[derive(Debug, Clone)]
pub enum WaveletNode {
    /// Internal node with a bit vector and two children.
    Internal {
        /// Bit vector marking right-half symbols.
        bv: BitVector,
        /// Left child ($\Sigma_L$).
        left: Box<WaveletNode>,
        /// Right child ($\Sigma_R$).
        right: Box<WaveletNode>,
    },
    /// Leaf node representing a single symbol.
    Leaf {
        /// The symbol value.
        symbol: u32,
    },
}

/// Wavelet Tree structure.
#[derive(Debug, Clone)]
pub struct WaveletTree {
    root: WaveletNode,
    len: usize,
    sigma: u32,
}

impl WaveletTree {
    /// Create a new Wavelet Tree from a sequence of symbols.
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
        let root = Self::build(data, 0, sigma);
        Self {
            root,
            len: data.len(),
            sigma,
        }
    }

    /// Return the alphabet size.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }

    fn build(data: &[u32], min: u32, max: u32) -> WaveletNode {
        if min + 1 >= max {
            return WaveletNode::Leaf { symbol: min };
        }

        let mid = min + (max - min) / 2;
        let mut bits = vec![0u64; data.len().div_ceil(64)];
        let mut left_data = Vec::new();
        let mut right_data = Vec::new();

        for (i, &v) in data.iter().enumerate() {
            if v >= mid {
                bits[i / 64] |= 1 << (i % 64);
                right_data.push(v);
            } else {
                left_data.push(v);
            }
        }

        let bv = BitVector::new(&bits, data.len());
        let left = Box::new(Self::build(&left_data, min, mid));
        let right = Box::new(Self::build(&right_data, mid, max));

        WaveletNode::Internal { bv, left, right }
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
        let mut curr = &self.root;
        while let WaveletNode::Internal { bv, left, right } = curr {
            if bv.get(i) {
                i = bv.rank1(i);
                curr = right;
            } else {
                i = bv.rank0(i);
                curr = left;
            }
        }
        if let WaveletNode::Leaf { symbol } = curr {
            *symbol
        } else {
            unreachable!("wavelet tree traversal ended at non-leaf node")
        }
    }

    /// Return the number of occurrences of `symbol` in the range [0, i).
    pub fn rank(&self, symbol: u32, mut i: usize) -> usize {
        let mut curr = &self.root;
        let mut min = 0;
        let mut max = self.sigma;

        while let WaveletNode::Internal { bv, left, right } = curr {
            let mid = min + (max - min) / 2;
            if symbol >= mid {
                i = bv.rank1(i);
                curr = right;
                min = mid;
            } else {
                i = bv.rank0(i);
                curr = left;
                max = mid;
            }
        }
        i
    }

    /// Return the position of the $k$-th occurrence of `symbol`.
    pub fn select(&self, symbol: u32, k: usize) -> Option<usize> {
        let pos = Self::select_recursive(&self.root, 0, self.sigma, symbol, k)?;
        if pos < self.len {
            Some(pos)
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
        Self::node_heap_bytes(&self.root)
    }

    fn node_heap_bytes(node: &WaveletNode) -> usize {
        match node {
            WaveletNode::Leaf { .. } => 0,
            WaveletNode::Internal { bv, left, right } => {
                bv.heap_bytes()
                    + std::mem::size_of::<WaveletNode>() * 2 // Box overhead
                    + Self::node_heap_bytes(left)
                    + Self::node_heap_bytes(right)
            }
        }
    }

    /// Serialize this wavelet tree to a stable binary encoding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"SBITWT01");
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&self.sigma.to_le_bytes());
        Self::serialize_node(&self.root, &mut out);
        out
    }

    fn serialize_node(node: &WaveletNode, out: &mut Vec<u8>) {
        match node {
            WaveletNode::Leaf { symbol } => {
                out.push(0u8); // tag: leaf
                out.extend_from_slice(&symbol.to_le_bytes());
            }
            WaveletNode::Internal { bv, left, right } => {
                out.push(1u8); // tag: internal
                let bv_bytes = bv.to_bytes();
                out.extend_from_slice(&(bv_bytes.len() as u64).to_le_bytes());
                out.extend_from_slice(&bv_bytes);
                Self::serialize_node(left, out);
                Self::serialize_node(right, out);
            }
        }
    }

    /// Deserialize a wavelet tree from `to_bytes()` output.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(bytes);
        r.read_magic(b"SBITWT01", "WaveletTree")?;
        let len = r.read_u64()? as usize;
        let sigma = r.read_u32()?;
        let root = Self::deserialize_node(&mut r)?;
        r.expect_eof("WaveletTree")?;
        Ok(Self { root, len, sigma })
    }

    fn deserialize_node(r: &mut ByteReader<'_>) -> Result<WaveletNode> {
        let tag = r.take(1)?[0];
        match tag {
            0 => {
                let symbol = r.read_u32()?;
                Ok(WaveletNode::Leaf { symbol })
            }
            1 => {
                let bv_len = r.read_u64()? as usize;
                let bv_bytes = r.take(bv_len)?;
                let bv = BitVector::from_bytes(bv_bytes)?;
                let left = Box::new(Self::deserialize_node(r)?);
                let right = Box::new(Self::deserialize_node(r)?);
                Ok(WaveletNode::Internal { bv, left, right })
            }
            _ => Err(Error::InvalidEncoding(format!(
                "WaveletTree: unknown node tag {tag}"
            ))),
        }
    }

    fn select_recursive(
        node: &WaveletNode,
        min: u32,
        max: u32,
        symbol: u32,
        k: usize,
    ) -> Option<usize> {
        match node {
            WaveletNode::Leaf { symbol: leaf_sym } => {
                if *leaf_sym == symbol {
                    Some(k)
                } else {
                    None
                }
            }
            WaveletNode::Internal { bv, left, right } => {
                let mid = min + (max - min) / 2;
                if symbol >= mid {
                    let pos = Self::select_recursive(right, mid, max, symbol, k)?;
                    bv.select1(pos)
                } else {
                    let pos = Self::select_recursive(left, min, mid, symbol, k)?;
                    bv.select0(pos)
                }
            }
        }
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
        // Use data where rank values are all different to avoid coincidental correctness.
        let data = vec![0, 0, 0, 1, 1, 2];
        let wt = WaveletTree::new(&data, 3);
        assert_eq!(wt.rank(0, 6), 3);
        assert_eq!(wt.rank(1, 6), 2);
        assert_eq!(wt.rank(2, 6), 1);
    }
}

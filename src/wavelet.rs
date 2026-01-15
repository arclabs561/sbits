//! Wavelet Tree for arbitrary alphabets.
//!
//! Generalizes rank and select operations from bit vectors to
//! sequences over larger alphabets $\Sigma$.
//!
//! # Theory
//!
//! A Wavelet Tree for a string $S$ of length $n$ over alphabet $\Sigma$:
//! - Root node partitions $\Sigma$ into two halves $\Sigma_L, \Sigma_R$.
//! - A bit vector at the root marks if $S[i] \in \Sigma_R$.
//! - Left child is Wavelet Tree for $S$ restricted to $\Sigma_L$.
//! - Right child is Wavelet Tree for $S$ restricted to $\Sigma_R$.
//!
//! Total space: $n \log |\Sigma| + o(n \log |\Sigma|)$ bits.
//! Queries `access`, `rank`, `select` take $O(\log |\Sigma|)$ time.

use crate::bitvec::BitVector;

/// Wavelet Tree node.
#[derive(Debug)]
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
#[derive(Debug)]
pub struct WaveletTree {
    root: WaveletNode,
    len: usize,
    sigma: u32,
}

impl WaveletTree {
    /// Create a new Wavelet Tree from a sequence of symbols.
    pub fn new(data: &[u32], sigma: u32) -> Self {
        let root = Self::build(data, 0, sigma);
        Self {
            root,
            len: data.len(),
            sigma,
        }
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
    pub fn access(&self, mut i: usize) -> u32 {
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
            0
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
        Self::select_recursive(&self.root, 0, self.sigma, symbol, k)
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
    }
}

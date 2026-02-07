//! # Succinct Data Structures
//!
//! *Near-optimal space with efficient queries.*
//!
//! ## Intuition First
//!
//! Imagine a library where every book is stored in a tightly packed vacuum-sealed bag.
//! It takes up the absolute minimum space (the information-theoretic limit).
//! Usually, to find a specific page, you'd have to unseal the whole bag (decompress).
//!
//! Succinct data structures allow you to "reach through" the vacuum seal and read
//! specific pages, count words, or find the k-th occurrence of a character—all
//! without unsealing the bag.
//!
//! ## The Problem
//!
//! Traditional data structures face a trade-off:
//! - **Pointers**: Fast queries ($O(1)$) but massive overhead ($O(n \log n)$ bits).
//! - **Compression**: Minimal space ($\log |C_n|$ bits) but no random access or queries ($O(n)$ to decompress).
//!
//! ## Historical Context
//!
//! ```text
//! 1971  Fano        Fano's associative memory: early partitioning of bits
//! 1974  Elias       Elias's static file storage: monotone sequences
//! 1989  Jacobson    Defined the succinct paradigm in his PhD thesis (rank/select)
//! 1996  Munro-Raman Constant-time rank and select in o(n) extra space
//! 2003  Grossi      Wavelet trees for rank/select over arbitrary alphabets
//! 2007  Ferragina   FM-index: compressed full-text search (the "BWT" era)
//! 2022  Kurpicz     Modern engineering of rank9/select9 structures
//! ```
//!
//! Guy Jacobson's key insight was that for many objects (like trees or bit vectors),
//! we can build auxiliary structures that are much smaller than the data itself ($o(n)$ bits)
//! but provide enough information to answer queries in constant time.
//!
//! ## Mathematical Formulation
//!
//! A data structure for a class of objects $C_n$ is:
//! - **Implicit**: Uses $\log |C_n| + O(1)$ bits.
//! - **Succinct**: Uses $\log |C_n| + o(\log |C_n|)$ bits.
//! - **Compact**: Uses $O(\log |C_n|)$ bits.
//!
//! The fundamental operations are:
//! - `rank(i)`: The number of set bits (1s) in the range $[0, i]$.
//! - `select(k)`: The position of the $k$-th set bit.
//!
//! ## Complexity Analysis
//!
//! - **Time**: $O(1)$ for rank/select in most modern implementations.
//! - **Space**: OPT + $o(n)$ bits (typically 5-20% overhead for auxiliary structures).
//!
//! ## What Could Go Wrong
//!
//! 1. **Static vs Dynamic**: Most succinct structures are static. Inserting into a
//!    succinct bit vector usually requires rebuilding the auxiliary structures.
//! 2. **Cache Locality**: Constant time $O(1)$ doesn't mean "fast" if it requires
//!    multiple random memory lookups for large bitsets.
//!
//! ## Implementation Notes
//!
//! This crate provides:
//! - **`BitVector`**: Core storage with rank/select support.
//! - **`WaveletTree`**: Rank/select over larger alphabets (planned).
//! - **`EliasFano`**: Monotone sequences with random access (planned).
//!
//! ## References
//!
//! - Jacobson, G. (1989). "Succinct Static Data Structures."
//! - Munro, J. I., & Raman, V. (1996). "Selection and counting on the fly."
//! - Grossi, R., et al. (2003). "High-order entropy-compressed text indexes."

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bitvec;
pub mod elias_fano;
pub mod error;
pub mod implicit;
pub mod partitioned_elias_fano;
pub mod rank_select;
pub mod wavelet;

pub use bitvec::BitVector;
pub use elias_fano::EliasFano;
pub use error::Error;
pub use partitioned_elias_fano::PartitionedEliasFano;
pub use wavelet::WaveletTree;

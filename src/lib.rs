//! Succinct data structures.
//!
//! This crate provides static structures that store bits, monotone integer
//! sequences, and symbol sequences compactly while preserving query operations.
//!
//! Provided structures:
//!
//! - [`BitVector`]: bit storage with rank/select support.
//! - [`EliasFano`]: monotone integer sequence with indexed access and
//!   successor/predecessor queries.
//! - [`PartitionedEliasFano`]: block-local Elias-Fano for clustered monotone
//!   sequences.
//! - [`WaveletTree`]: rank/select over larger alphabets.
//!
//! These structures are static. Updating values usually means rebuilding the
//! structure.
//!
//! References:
//!
//! - Jacobson, G. (1989). "Succinct Static Data Structures."
//! - Munro, J. I., & Raman, V. (1996). "Selection and counting on the fly."
//! - Grossi, R., et al. (2003). "High-order entropy-compressed text indexes."

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod bitvec;
pub mod elias_fano;
pub mod error;
pub mod partitioned_elias_fano;
pub mod wavelet;

pub use bitvec::BitVector;
pub use elias_fano::EliasFano;
pub use error::Error;
pub use partitioned_elias_fano::PartitionedEliasFano;
pub use wavelet::WaveletTree;

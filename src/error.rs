//! Error types for succinct data structures.

use thiserror::Error;

/// Error variants for succinct data structure operations.
#[derive(Debug, Error)]
pub enum Error {
    /// An index was provided that is out of the structure's bounds.
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),

    /// A selection query was performed for a rank that does not exist.
    #[error("invalid selection: rank {0} not found")]
    InvalidSelection(usize),

    /// An I/O error occurred during serialization or deserialization.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// A specialized Result type for succinct operations.
pub type Result<T> = std::result::Result<T, Error>;

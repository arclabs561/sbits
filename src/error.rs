//! Error types and deserialization utilities for succinct data structures.

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

    /// Invalid binary encoding for a succinct structure.
    #[error("invalid encoding: {0}")]
    InvalidEncoding(String),
}

/// A specialized Result type for succinct operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Cursor for reading little-endian binary encodings.
pub(crate) struct ByteReader<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, off: 0 }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.off + n > self.data.len() {
            return Err(Error::InvalidEncoding(
                "unexpected end of input".to_string(),
            ));
        }
        let slice = &self.data[self.off..self.off + n];
        self.off += n;
        Ok(slice)
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn read_magic(&mut self, expected: &[u8; 8], name: &str) -> Result<()> {
        let magic = self.take(8)?;
        if magic != expected {
            return Err(Error::InvalidEncoding(format!("bad magic for {name}")));
        }
        Ok(())
    }

    /// Check that a claimed array length doesn't exceed input size (prevents allocation bombs).
    pub fn check_alloc(&self, count: usize, item_bytes: usize) -> Result<()> {
        if count.saturating_mul(item_bytes) > self.data.len() {
            return Err(Error::InvalidEncoding(format!(
                "claimed length ({count}) too large for input ({} bytes)",
                self.data.len()
            )));
        }
        Ok(())
    }

    pub fn read_u64_vec(&mut self, count: usize) -> Result<Vec<u64>> {
        self.check_alloc(count, 8)?;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(self.read_u64()?);
        }
        Ok(v)
    }

    pub fn read_u32_vec(&mut self, count: usize) -> Result<Vec<u32>> {
        self.check_alloc(count, 4)?;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(self.read_u32()?);
        }
        Ok(v)
    }

    pub fn expect_eof(&self, name: &str) -> Result<()> {
        if self.off != self.data.len() {
            return Err(Error::InvalidEncoding(format!(
                "trailing bytes after {name}"
            )));
        }
        Ok(())
    }
}

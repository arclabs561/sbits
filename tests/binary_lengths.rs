//! Nested byte lengths are untrusted and must return errors, never panic.

use sbits::{EliasFano, PartitionedEliasFano, WaveletTree};

#[test]
fn elias_fano_rejects_overflowing_upper_bitvector_length() {
    let values = [0, 1, 2];
    let mut bytes = EliasFano::new(&values, 3).to_bytes();
    assert_eq!(
        EliasFano::from_bytes(&bytes)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        values
    );
    // With l = 0, there are no lower words; upper byte length follows lower_len.
    bytes[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(EliasFano::from_bytes(&bytes).is_err());
}

#[test]
fn partitioned_elias_fano_rejects_overflowing_block_length() {
    let values = [10, 20];
    let mut bytes = PartitionedEliasFano::new(&values, 100, 2).to_bytes();
    assert_eq!(
        PartitionedEliasFano::from_bytes(&bytes)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        values
    );
    // The first block's byte length follows the fixed header and one base.
    bytes[44..52].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(PartitionedEliasFano::from_bytes(&bytes).is_err());
}

#[test]
fn wavelet_tree_rejects_overflowing_level_length() {
    let mut bytes = WaveletTree::new(&[0, 1], 2).to_bytes();
    let decoded = WaveletTree::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.access(0), 0);
    assert_eq!(decoded.access(1), 1);
    // The first level's byte length follows magic, len, sigma and depth.
    bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(WaveletTree::from_bytes(&bytes).is_err());
}

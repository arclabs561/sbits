//! `from_bytes` must reject encodings that `new` could never produce.

use sbits::{EliasFano, PartitionedEliasFano, WaveletTree};

/// `to_bytes` layout of a `PartitionedEliasFano` with the given blocks.
fn pef_bytes(universe: u64, block_size: u32, n: u64, blocks: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"SBITPEF2");
    out.extend_from_slice(&universe.to_le_bytes());
    out.extend_from_slice(&block_size.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&(blocks.len() as u64).to_le_bytes());
    for (base, _) in blocks {
        out.extend_from_slice(&base.to_le_bytes());
    }
    for (_, b) in blocks {
        out.extend_from_slice(&(b.len() as u64).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

#[test]
fn elias_fano_rejects_decoded_value_outside_universe() {
    let ef = EliasFano::new(&[9], 10); // l = 3: low bits 1, high bits 1
    let mut bytes = ef.to_bytes();
    // magic 8 + universe 8 + l 4 + n 8 + lower_len 8 = 36: first lower word.
    assert_eq!(bytes[36], 1);
    bytes[36] = 7; // now decodes to 15, outside universe 10
    assert!(EliasFano::from_bytes(&bytes).is_err());
}

#[test]
fn elias_fano_rejects_decoded_values_out_of_order() {
    let ef = EliasFano::new(&[9, 10], 16); // l = 3: low bits 1 and 2, same high bucket
    let mut bytes = ef.to_bytes();
    // Lower bits are packed little-endian, 3 bits each: value 0 then value 1.
    assert_eq!(bytes[36], 1 | (2 << 3));
    bytes[36] = 7 | (1 << 3); // decodes to [15, 9]
    assert!(EliasFano::from_bytes(&bytes).is_err());
}

#[test]
fn round_trip_still_loads() {
    let values = [0, 3, 3, 17, 255, 1000];
    let ef = EliasFano::new(&values, 1001);
    let back = EliasFano::from_bytes(&ef.to_bytes()).unwrap();
    assert_eq!(back.iter().collect::<Vec<_>>(), values);
    let pef = PartitionedEliasFano::new(&values, 1001, 4);
    let back = PartitionedEliasFano::from_bytes(&pef.to_bytes()).unwrap();
    assert_eq!(back.iter().collect::<Vec<_>>(), values);
}

#[test]
fn partitioned_rejects_blocks_that_disagree_with_block_size() {
    // block_size 2, but the first block holds 1 value and the second 3, so
    // get(i), which maps i to (i / 2, i % 2), would disagree with iter().
    let b0 = EliasFano::new(&[0], 1).to_bytes();
    let b1 = EliasFano::new(&[0, 1, 2], 3).to_bytes();
    let bytes = pef_bytes(100, 2, 4, &[(10, b0), (20, b1)]);
    assert!(PartitionedEliasFano::from_bytes(&bytes).is_err());
}

#[test]
fn wavelet_rejects_symbols_not_below_sigma() {
    let wt = WaveletTree::new(&[3, 3, 0], 4);
    let mut bytes = wt.to_bytes();
    // Header: magic (8), len (8), sigma (4). Lowering sigma to 3 keeps depth 2,
    // so the levels still decode symbol 3, which `new` rejects for sigma 3.
    assert_eq!(bytes[16..20], 4u32.to_le_bytes());
    bytes[16..20].copy_from_slice(&3u32.to_le_bytes());
    assert!(WaveletTree::from_bytes(&bytes).is_err());

    let ok = WaveletTree::new(&[2, 2, 0, 1], 3);
    let back = WaveletTree::from_bytes(&ok.to_bytes()).unwrap();
    assert_eq!(
        (0..4).map(|i| back.access(i)).collect::<Vec<_>>(),
        [2, 2, 0, 1]
    );
}

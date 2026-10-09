//! Deserializing through serde must apply the same checks as `from_bytes`.
//!
//! The rank/select directory, the Elias-Fano counts and the wavelet zero
//! boundaries are derived data. `from_bytes` rejects them when they disagree
//! with the stored bits; a serde round trip through tampered data must be
//! rejected the same way instead of building a structure that answers queries
//! wrongly or indexes out of bounds.
#![cfg(feature = "serde")]

use sbits::{BitVector, EliasFano, PartitionedEliasFano, WaveletTree};
use serde_json::{json, Value};

fn bump_first(value: &mut Value, field: &str) {
    let arr = value[field].as_array_mut().expect("array field");
    let first = arr[0].as_u64().unwrap();
    arr[0] = json!(first + 1);
}

#[test]
fn bitvector_rejects_a_tampered_select_index() {
    let bits = [0xF0F0_F0F0_F0F0_F0F0u64; 20];
    let bv = BitVector::new(&bits, 20 * 64);
    let good = serde_json::to_value(&bv).unwrap();
    assert!(serde_json::from_value::<BitVector>(good.clone()).is_ok());

    let mut bad = good;
    bump_first(&mut bad, "select1_index");
    assert!(serde_json::from_value::<BitVector>(bad).is_err());
}

#[test]
fn elias_fano_rejects_a_tampered_count() {
    let ef = EliasFano::new(&[3, 9, 27, 81, 243], 1000);
    let good = serde_json::to_value(&ef).unwrap();
    assert!(serde_json::from_value::<EliasFano>(good.clone()).is_ok());

    let mut bad = good;
    bad["n"] = json!(6);
    assert!(serde_json::from_value::<EliasFano>(bad).is_err());
}

#[test]
fn partitioned_elias_fano_rejects_a_tampered_count() {
    let values: Vec<u64> = (0..50).map(|i| i * 3).collect();
    let pef = PartitionedEliasFano::new(&values, 1000, 8);
    let good = serde_json::to_value(&pef).unwrap();
    assert!(serde_json::from_value::<PartitionedEliasFano>(good.clone()).is_ok());

    let mut bad = good.clone();
    bad["n"] = json!(51);
    assert!(serde_json::from_value::<PartitionedEliasFano>(bad).is_err());

    let mut bad = good;
    bad["bases"].as_array_mut().unwrap().pop();
    assert!(serde_json::from_value::<PartitionedEliasFano>(bad).is_err());
}

#[test]
fn wavelet_tree_rejects_tampered_zero_boundaries() {
    let wt = WaveletTree::new(&[3, 1, 2, 0, 3, 0, 1, 2], 4);
    let good = serde_json::to_value(&wt).unwrap();
    assert!(serde_json::from_value::<WaveletTree>(good.clone()).is_ok());

    let mut bad = good;
    bump_first(&mut bad, "zeros");
    assert!(serde_json::from_value::<WaveletTree>(bad).is_err());
}

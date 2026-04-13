# sbits

[![crates.io](https://img.shields.io/crates/v/sbits.svg)](https://crates.io/crates/sbits)
[![Documentation](https://docs.rs/sbits/badge.svg)](https://docs.rs/sbits)
[![CI](https://github.com/arclabs561/sbits/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sbits/actions/workflows/ci.yml)

Succinct data structures.

Dual-licensed under MIT or Apache-2.0.

## Structures

| Structure | What it does |
|-----------|-------------|
| `BitVector` | Rank/select over bits in O(1)/O(log n) with Rank9 interleaved layout |
| `EliasFano` | Sorted integers in near-optimal space with O(1) access, O(1) avg successor |
| `PartitionedEliasFano` | Block-local Elias-Fano for clustered sequences |
| `WaveletTree` | Rank/select over arbitrary alphabets in O(log sigma) |

## Quickstart

```toml
[dependencies]
sbits = "0.1.2"
```

```rust
use sbits::bitvec::BitVector;
use sbits::elias_fano::EliasFano;

// BitVector: rank, select, iterate
let bv = BitVector::from_ones([0, 1, 3, 7].into_iter(), 64);
assert_eq!(bv.rank1(4), 3);
assert_eq!(bv.select1(2), Some(3));
let ones: Vec<usize> = bv.ones().collect();
assert_eq!(ones, vec![0, 1, 3, 7]);

// Elias-Fano: compressed sorted integers with successor/predecessor
let ef = EliasFano::new(&[10, 20, 30, 100, 1000], 2000);
assert_eq!(ef.successor(15), Some(20));
assert_eq!(ef.predecessor(25), Some(20));
assert!(ef.contains(30));

// Iterate all values
let all: Vec<u32> = ef.iter().collect();
assert_eq!(all, vec![10, 20, 30, 100, 1000]);
```

## Serialization

All structures support stable binary serialization:

```rust
use sbits::EliasFano;

let ef = EliasFano::new(&[10, 20, 30], 100);
let bytes = ef.to_bytes();
let ef2 = EliasFano::from_bytes(&bytes).unwrap();
assert_eq!(ef2.get(0).unwrap(), 10);
```

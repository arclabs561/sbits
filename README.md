# sbits

[![crates.io](https://img.shields.io/crates/v/sbits.svg)](https://crates.io/crates/sbits)
[![Documentation](https://docs.rs/sbits/badge.svg)](https://docs.rs/sbits)
[![CI](https://github.com/arclabs561/sbits/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sbits/actions/workflows/ci.yml)

Succinct data structures.

Dual-licensed under MIT or Apache-2.0.

## Operations

| Operation | Definition |
|-----------|-----------|
| rank₁(*k*) | Number of 1-bits in positions \[0, *k*) |
| select₁(*k*) | Position of the *k*-th 1-bit (0-indexed) |
| Elias-Fano | Encodes *m* sorted integers from \[0, *n*) in 2*m* + *m*⌈log₂(*n*/*m*)⌉ bits |

## Quickstart

```toml
[dependencies]
sbits = "0.1.2"
```

```rust
use sbits::bitvec::BitVector;
use sbits::elias_fano::EliasFano;

let bv = BitVector::new(&[0b1011], 64);
assert_eq!(bv.rank1(4), 3);
assert_eq!(bv.select1(2), Some(3));

// Iterate over set bits
let ones: Vec<usize> = bv.ones().collect();
assert_eq!(ones, vec![0, 1, 3]);

// Elias-Fano with predecessor/successor
let ef = EliasFano::new(&[10, 20, 30, 100, 1000], 2000);
assert_eq!(ef.successor(15), Some(20));
assert_eq!(ef.predecessor(25), Some(20));
```

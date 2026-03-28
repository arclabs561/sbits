# sbits

[![crates.io](https://img.shields.io/crates/v/sbits.svg)](https://crates.io/crates/sbits)
[![Documentation](https://docs.rs/sbits/badge.svg)](https://docs.rs/sbits)
[![CI](https://github.com/arclabs561/sbits/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sbits/actions/workflows/ci.yml)

Succinct data structures: near-optimal space with efficient queries.

Dual-licensed under MIT or Apache-2.0.

## Operations

| Operation | Definition |
|-----------|-----------|
| $\text{rank}_1(k)$ | Number of 1-bits in positions $[0, k)$ |
| $\text{select}_1(k)$ | Position of the $k$-th 1-bit (0-indexed) |
| Elias-Fano | Encodes $m$ sorted integers from $[0, n)$ in $2m + m\lceil\log_2(n/m)\rceil$ bits |

## Quickstart

```toml
[dependencies]
sbits = "0.1.1"
```

```rust
use sbits::bitvec::BitVector;
use sbits::elias_fano::EliasFano;

let bv = BitVector::new(&[0b1011], 64);
assert_eq!(bv.rank1(4), 3);
assert_eq!(bv.select1(2), Some(3));
```

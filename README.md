# sbits

[![Documentation](https://docs.rs/sbits/badge.svg)](https://docs.rs/sbits)

Succinct data structures: near-optimal space with efficient queries.

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
sbits = "0.1.0"
```

```rust
use sbits::bitvec::BitVector;
use sbits::elias_fano::EliasFano;

let bv = BitVector::new(&[0b1011], 64);
assert_eq!(bv.rank1(4), 3);
assert_eq!(bv.select1(2), Some(3));
```

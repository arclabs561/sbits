# succ

Succinct data structures: near-optimal space with efficient queries.

Dual-licensed under MIT or the UNLICENSE.

```rust
use succ::bitvec::BitVector;
use succ::elias_fano::EliasFano;

let bv = BitVector::new(&[0b1011], 64);
assert_eq!(bv.rank1(4), 3);
assert_eq!(bv.select1(2), Some(3));
```

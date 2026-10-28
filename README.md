# sbits

[![crates.io](https://img.shields.io/crates/v/sbits.svg)](https://crates.io/crates/sbits)
[![Documentation](https://docs.rs/sbits/badge.svg)](https://docs.rs/sbits)
[![CI](https://github.com/arclabs561/sbits/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sbits/actions/workflows/ci.yml)

Succinct data structures.

Store bits, sorted integers or symbol sequences compactly and query them without
decompressing the whole structure. Rank counts occurrences before a position;
select locates the k-th occurrence. The structures are static: changing a value
means rebuilding.

Dual-licensed under MIT or Apache-2.0.

## Structures

| Structure | What it does |
|-----------|-------------|
| `BitVector` | Rank/select over bits in O(1)/O(log n) with Rank9 interleaved layout |
| `EliasFano` | Compressed sorted integers with indexed access and successor queries |
| `PartitionedEliasFano` | Block-local Elias-Fano for clustered sequences |
| `WaveletTree` | Rank, select and indexed access over symbol sequences |

## Quickstart

```toml
[dependencies]
sbits = "0.2"
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
let all: Vec<u64> = ef.iter().collect();
assert_eq!(all, vec![10, 20, 30, 100, 1000]);
```

```rust
use sbits::WaveletTree;

// A sequence over a small alphabet (0 = INFO, 1 = WARN, 2 = ERROR).
let levels = [0u32, 0, 2, 1, 0, 2, 2, 1, 0, 2];
let wt = WaveletTree::new(&levels, 3);

assert_eq!(wt.access(2), 2);          // symbol at index 2
assert_eq!(wt.rank(2, 7), 3);         // ERROR codes in [0, 7)
assert_eq!(wt.select(2, 0), Some(2)); // index of the first ERROR
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

## Feature flags

The default `std` feature enables standard-library integration. Disable default
features for `no_std` with `alloc`. Enable `serde` for Serde serialization.

## Examples

Runnable examples live in [`examples/`](examples/):

- `sorted_integers` packs a simulated posting list (1000 doc IDs out of a 1M-document corpus) with Elias-Fano.
- `wavelet_sequence` queries a sequence drawn from a small alphabet.

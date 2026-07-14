# sbits

Succinct data structures -- bit vector rank/select, Elias-Fano compressed integer sequences, and wavelet trees -- with near-optimal space and constant or logarithmic query time. Backed by a Rust implementation.

Python bindings for the [sbits](https://crates.io/crates/sbits) Rust crate.

## Install

    pip install sbits

## Quick start

```python
from sbits import BitVector, EliasFano, WaveletTree

# Bit vector with O(1) rank and O(log n) select
bv = BitVector([True, False, True, True, False, True])
bv.rank(4)    # 3 -- number of set bits in [0, 4)
bv.select(2)  # 3 -- position of the 3rd set bit (0-indexed)
bv.get(0)     # True

# Elias-Fano compressed sorted integers with O(1) access
ef = EliasFano([10, 20, 30, 100, 1000])
ef[0]         # 10
ef[2]         # 30
30 in ef      # True

# Wavelet tree: rank/select over arbitrary integer alphabets
wt = WaveletTree([3, 1, 2, 0, 3, 0, 1, 2], sigma=4)
wt.access(0)     # 3
wt.rank(3, 8)    # 2 -- occurrences of 3 in [0, 8)
wt.select(3, 0)  # 0 -- position of first 3
```

## API

| Name | Description |
|------|-------------|
| `BitVector(bits)` | Construct from booleans, supports rank/select/get |
| `BitVector.rank(i)` | Count set bits in [0, i), O(1) |
| `BitVector.select(k)` | Position of k-th set bit (0-indexed), O(log n) |
| `BitVector.get(i)` | Test whether bit i is set |
| `BitVector.count_ones()` | Total number of set bits |
| `BitVector.to_numpy()` | Export as numpy boolean array |
| `EliasFano(values)` | Construct from sorted non-negative integers |
| `EliasFano.get(i)` | Value at index i, O(1) |
| `EliasFano.contains(v)` | Membership test, O(log n) |
| `EliasFano.successor(v)` | Smallest value >= v, O(log bucket) |
| `EliasFano.predecessor(v)` | Largest value <= v, O(log bucket) |
| `EliasFano.to_numpy()` | Export as numpy uint64 array |
| `WaveletTree(data, sigma)` | Construct from symbol sequence and alphabet size |
| `WaveletTree.rank(sym, i)` | Count occurrences of sym in [0, i), O(log sigma) |
| `WaveletTree.select(sym, k)` | Position of k-th occurrence of sym, O(log sigma) |
| `WaveletTree.access(i)` | Symbol at position i, O(log sigma) |

## numpy support

Constructors accept numpy arrays or Python lists. Export methods (`to_numpy`) return numpy arrays. `BitVector` and `EliasFano` support `len()`, indexing, and `in`.

## License

MIT OR Apache-2.0

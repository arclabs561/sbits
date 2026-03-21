# sbits

Python bindings for the [sbits](https://crates.io/crates/sbits) Rust crate.

Provides succinct data structures with near-optimal space and efficient queries:

- **BitVector** -- rank and select in O(1) time with o(n) auxiliary space.
- **EliasFano** -- compressed sorted integer sequences with O(1) random access.
- **WaveletTree** -- rank, select, and access over arbitrary alphabets in O(log sigma) time.

## Installation

```
pip install sbits
```

## Usage

```python
from sbits import BitVector, EliasFano, WaveletTree

# Bit vector with rank/select
bv = BitVector([True, False, True, True, False, True])
bv.rank(4)    # 3 (number of set bits in [0, 4))
bv.select(2)  # 3 (position of 3rd set bit, 0-indexed)

# Elias-Fano for sorted integers
ef = EliasFano([10, 20, 30, 100, 1000])
ef[0]         # 10
30 in ef      # True

# Wavelet tree
wt = WaveletTree([3, 1, 2, 0, 3, 0, 1, 2], sigma=4)
wt.access(0)     # 3
wt.rank(3, 8)    # 2
wt.select(3, 0)  # 0
```

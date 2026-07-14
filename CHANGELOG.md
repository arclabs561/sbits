# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Tightened crate-level docs around the public data-structure contract.
- Aligned the Python Elias-Fano bindings with the 64-bit core API. NumPy exports now use `uint64`; `uint32` inputs remain accepted.

## [0.2.3] - 2026-06-11

Maintenance release.

## [0.2.2] - 2026-04-20

### Added

- Add sorted_integers example (Elias-Fano posting list)
- Add comparison benchmarks vs sucds and vers-vecs

### Changed

- Cursor-based EF iterator, select lookup table, shared wavelet rank
- Further select optimization
- Further rank/select optimization
- Optimize successor/rank with direct upper-bit scanning

## [0.2.0] - 2026-04-13

### Changed

- No_std + serde feature
- U64 EliasFano, WaveletMatrix layout (breaking)
- Replace WaveletTree internals with WaveletMatrix layout
- Update Python bindings for v0.1.3

## [0.1.3] - 2026-04-13

### Changed

- Zero dependencies, bounds-checked access
- Input validation, completeness, comprehensive benchmarks
- Broadword select experiment -- reverted, sequential wins on ARM
- Iterators, predecessor/successor, select speedup, test coverage

## [0.1.2] - 2026-04-06

### Added

- Add math definitions for rank, select, and Elias-Fano
- Add Python bindings
- Add doctests for BitVector and EliasFano
- Serialization, partitioned Elias-Fano, l=0 edge case fix

### Changed

- Validate sorted order in EliasFano constructor
- Expand API
- API polish
- Bindings accept numpy arrays, add to_numpy()
- Initial import

from typing import Iterator, Optional, Union

import numpy as np
from numpy.typing import NDArray

__version__: str

class BitVector:
    """Succinct bit vector with O(1) rank and select.

    Space: n + o(n) bits (near-optimal with small auxiliary structures).
    """

    def __init__(self, bits: Union[NDArray[np.bool_], list[bool]]) -> None:
        """Construct from a sequence of booleans.

        Args:
            bits: Boolean values as numpy array or list.
        """
        ...

    def rank(self, i: int) -> int:
        """Count set bits in [0, i). O(1) time."""
        ...

    def select(self, k: int) -> Optional[int]:
        """Position of the k-th set bit (0-indexed). O(1) time.

        Returns None if k >= count_ones.
        """
        ...

    def get(self, i: int) -> bool:
        """Return True if the bit at index i is set."""
        ...

    def count_ones(self) -> int:
        """Number of set bits in the entire vector."""
        ...

    def to_numpy(self) -> NDArray[np.bool_]:
        """Export all bits as a numpy boolean array."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> bool: ...
    def __iter__(self) -> Iterator[bool]: ...
    def __repr__(self) -> str: ...

class EliasFano:
    """Elias-Fano encoding for sorted sequences of non-negative integers.

    Space: 2n + n*ceil(log2(u/n)) + o(n) bits, where u is the universe size.
    """

    def __init__(
        self, values: Union[NDArray[np.uint32], NDArray[np.int64], list[int]]
    ) -> None:
        """Construct from a sorted sequence of non-negative integers.

        Args:
            values: Sorted values as numpy array (uint32 or int64) or list.
        """
        ...

    def get(self, i: int) -> int:
        """Return the value at index i. O(1) time."""
        ...

    def contains(self, value: int) -> bool:
        """Return True if the value is present (linear scan)."""
        ...

    def to_numpy(self) -> NDArray[np.uint32]:
        """Export all values as a numpy uint32 array."""
        ...

    def len(self) -> int: ...
    def __len__(self) -> int: ...
    def __contains__(self, value: int) -> bool: ...
    def __getitem__(self, i: int) -> int: ...
    def __repr__(self) -> str: ...

class WaveletTree:
    """Wavelet tree for rank/select over arbitrary integer alphabets.

    Space: n*ceil(log2(sigma)) + o(n*log(sigma)) bits.
    """

    def __init__(
        self, data: Union[NDArray[np.uint32], list[int]], sigma: int
    ) -> None:
        """Construct from a sequence and alphabet size.

        Args:
            data: Symbol sequence as numpy uint32 array or list.
            sigma: Alphabet size (number of distinct symbols).
        """
        ...

    def rank(self, symbol: int, i: int) -> int:
        """Count occurrences of symbol in [0, i). O(log(sigma)) time."""
        ...

    def select(self, symbol: int, k: int) -> Optional[int]:
        """Position of the k-th occurrence of symbol (0-indexed). O(log(sigma)) time.

        Returns None if fewer than k+1 occurrences exist.
        """
        ...

    def access(self, i: int) -> int:
        """Return the symbol at index i. O(log(sigma)) time."""
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> int: ...
    def __repr__(self) -> str: ...

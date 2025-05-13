"""This module contains the ``Freq`` class."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsInt

from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import Serializable


@dataclass(init=False)
class Freq(Serializable):
    """Class containing the frequency data of a result returned by a solver."""

    bitvectors: list[BitVector]
    """List with measured bitstrings as :py:class:`~tno.quantum.utils.BitVector`."""
    energies: list[float]
    """List containing energies as float."""
    num_occurrences: list[int]
    """List containing the number of occurrences as int."""

    def __init__(
        self,
        bitvectors: Iterable[BitVectorLike],
        energies: Iterable[SupportsFloat],
        num_occurrences: Iterable[SupportsInt],
    ) -> None:
        """Init of :py:class:`Freq`.

        Args:
            bitvectors: Iterable over bitvectors. All values in `bitvectors` should be
                safely castable to :py:class:`~tno.quantum.utils.BitVector`.
            energies: Iterable over energies. All values in `energies` should
                be safely castable to float.
            num_occurrences: Iterable over number of occurrences. All values in
                `num_occurrences` should be safely castable to int.

        Raises:
            TypeError: If elements in `bitvectors`, `energies` or `num_occurrences`
                cannot be cast to :py:class:`~tno.quantum.utils.BitVector`, float or
                int, respectively.
            ValueError: If `bitvectors`, `energies` and `num_occurrences` do not contain
                the same number of elements.
        """
        self.bitvectors = [BitVector(b) for b in bitvectors]
        self.energies = [float(e) for e in energies]
        self.num_occurrences = [int(n) for n in num_occurrences]

        n = len(self.bitvectors)
        if n != len(self.energies) or n != len(self.bitvectors):
            msg = "`bitvectors`, `energies` and `num_occurrences` must have same length"
            raise ValueError(msg)

    def __iter__(self) -> Iterator[tuple[BitVector, float, int]]:
        """Iterator over elements of `bitvectors`, `energies` and `num_occurrences`."""
        return zip(self.bitvectors, self.energies, self.num_occurrences)

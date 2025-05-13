"""This module contains the ``BasicResult`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat

from tno.quantum.optimization.qubo.components._freq import Freq
from tno.quantum.optimization.qubo.components._results._result_interface import (
    ResultInterface,
)
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.validation import check_real

if TYPE_CHECKING:
    from typing import Self


class BasicResult(ResultInterface):
    """Most basic implementation of :py:class:`ResultInterface`."""

    @classmethod
    def from_result(
        cls,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq | None = None,
    ) -> Self:
        """Create :py:class:`BasicResult` from bitvector, value and frequency object."""
        best_bitvector = BitVector(best_bitvector)
        best_value = check_real(best_value, "best_value")
        if freq is None:
            freq = Freq([best_bitvector], [best_value], [1])

        return cls(best_bitvector, best_value, freq)

    def __eq__(self, other: Any) -> bool:
        """Check if two instances of :py:class:`BasicResult` are equal."""
        if not isinstance(other, BasicResult):
            return False

        return (
            self.best_value == other.best_value
            and self.best_bitvector == other.best_bitvector
            and self.freq == other.freq
        )

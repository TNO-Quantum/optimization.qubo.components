"""Generic solver functionality for a QUBO solver."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, Generic, TypeVar

from tno.quantum.optimization.qubo.components._results._result_interface import (
    ResultInterface,
)

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components._qubo import QUBO

RESULT_TYPE = TypeVar("RESULT_TYPE", bound=ResultInterface)


class Solver(ABC, Generic[RESULT_TYPE]):
    """Abstract QUBO solver base class."""

    non_deterministic: bool = False
    """Flag that indicates whether the solver is non-deterministic given a fixed random
    state."""

    def solve(self, qubo: QUBO) -> RESULT_TYPE:
        """Solve the given QUBO.

        Args:
            qubo: QUBO to solve.

        Returns:
            :py:class:`ResultInterface` containing the result of the solve. Note that
            the implementation of the :py:class:`ResultInterface` class depends on the
            solver.
        """
        start_time = time.perf_counter()
        result = self._solve(qubo)
        end_time = time.perf_counter()

        # Set execution time on result
        execution_time = end_time - start_time
        result.execution_time = timedelta(seconds=execution_time)

        return result

    @abstractmethod
    def _solve(self, qubo: QUBO) -> RESULT_TYPE:
        """Solve the given QUBO.

        Args:
            qubo: QUBO to solve.

        Returns:
            ``ResultInterface`` containing the result of the solve.
        """

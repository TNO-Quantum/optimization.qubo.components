"""This module contains the ``PostprocessorConfig`` configuration class."""

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


RETURN_TYPE = TypeVar("RETURN_TYPE", bound=ResultInterface)


class Postprocessor(ABC, Generic[RETURN_TYPE]):
    """Abstract postprocessor base class.

    Postprocessors can be used to improve the result obtained by a :py:class:`Solver`.
    """

    def postprocess(self, qubo: QUBO, hint: ResultInterface) -> RETURN_TYPE:
        """Performs postprocessing on the given result with respect to the given QUBO.

        Args:
            qubo: QUBO to be solved.
            hint: Result from an earlier solve, used as starting point for the
                postprocessor.

        Returns:
            A postprocessed result.
        """
        start_time = time.perf_counter()
        result = self._postprocess(qubo, hint)
        end_time = time.perf_counter()

        # Set execution time on result
        execution_time = end_time - start_time
        result.execution_time = timedelta(seconds=execution_time)

        return result

    @abstractmethod
    def _postprocess(self, qubo: QUBO, hint: ResultInterface) -> RETURN_TYPE:
        """Performs postprocessing on the given result with respect to the given QUBO.

        Args:
            qubo: QUBO to be solved.
            hint: Result from an earlier solve, used as starting point for the
                postprocessor.

        Returns:
            A postprocessed result.
        """

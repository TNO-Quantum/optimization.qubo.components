"""Module that contains the `Preprocessor` class."""

from abc import ABC, abstractmethod

from tno.quantum.optimization.qubo.components._preprocessing._partial_solution import (
    PartialSolution,
)
from tno.quantum.optimization.qubo.components._qubo import QUBO


class Preprocessor(ABC):
    """Abstract preprocessor base class.

    Preprocessors can be used to find partial solutions to a QUBO problem, allowing
    to reduce the number of variables of a QUBO while maintaining an optimal solution.
    """

    @abstractmethod
    def preprocess(self, qubo: QUBO) -> tuple[PartialSolution, QUBO]:
        """Performs preprocessing on the given QUBO.

        Args:
            qubo: QUBO to be preprocessed.

        Returns:
            Partial solution and corresponding preprocessed QUBO.
        """

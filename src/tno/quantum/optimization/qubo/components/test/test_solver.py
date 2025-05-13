"""Tests for the ``Solver`` base class and the ``check_random_state`` method."""

from __future__ import annotations

import numpy as np
import pytest

from tno.quantum.optimization.qubo.components import QUBO, BasicResult, Freq, Solver

QUBO_BEST_VALUE = -10


@pytest.fixture(name="qubo")
def qubo_fixture() -> QUBO:
    """Simple test QUBO"""
    return QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])


class TestNonDeterministicWarning:
    """Test solver with that is non deterministic."""

    @pytest.fixture(name="non_deterministic_solver")
    def non_deterministic_solver_fixture(self) -> Solver[BasicResult]:
        """Simple solver that has non deterministic flag set to True."""

        class NonDeterministicSolver(Solver[BasicResult]):
            """Test Solver"""

            non_deterministic: bool = True

            def _solve(self, qubo: QUBO) -> BasicResult:
                """

                Args:
                    qubo: QUBO formulation to solve.
                    random_state: random state.

                Returns:
                    random solution.
                """
                # random input info, which passes validation checks
                rng = np.random.default_rng()
                bits = [rng.integers(0, 1) for _ in range(len(qubo))]
                freq = Freq([bits], [0], [1])
                return BasicResult.from_result(bits, qubo.evaluate(bits), freq)

        return NonDeterministicSolver()

    def test_non_deterministic_solver(
        self, non_deterministic_solver: Solver[BasicResult]
    ) -> None:
        """Test non_deterministic property."""
        assert non_deterministic_solver.non_deterministic is True

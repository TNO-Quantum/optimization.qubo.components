"""Test get solver functionality"""

import pytest

from tno.quantum.optimization.qubo.components import (
    QUBO,
    BasicResult,
    Freq,
    Solver,
    SolverConfig,
)


class DummySolver(Solver[BasicResult]):
    def _solve(self, qubo: QUBO) -> BasicResult:
        return BasicResult(best_bitvector="0", best_value=0, freq=Freq([], [], []))


SolverConfig.register_custom_item("dummy_solver", DummySolver)


def test_solver_config_supported_items_contains_dummy_solver() -> None:
    """Test ``SolverConfig.supported_items()`` contains ``DummySolver``."""
    supported_solvers = SolverConfig.supported_custom_items()
    assert len(supported_solvers) > 0
    assert "dummy_solver" in supported_solvers
    assert supported_solvers["dummy_solver"] is DummySolver


def test_solver_config_get_instance() -> None:
    """Test `get_instance` from SolverConfig."""
    solver_config = SolverConfig(
        name="dummy_solver",
        options={},
    )
    solver = solver_config.get_instance()
    assert isinstance(solver, DummySolver)


def test_get_invalid_solver() -> None:
    """Test get solver for invalid input."""
    with pytest.raises(KeyError, match="does not match any of the supported items"):
        SolverConfig(name="invalid_solver_name")

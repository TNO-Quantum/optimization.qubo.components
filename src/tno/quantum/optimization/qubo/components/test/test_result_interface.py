"""This module contains tests for the ``ResultInterface``."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from tno.quantum.optimization.qubo.components import BasicResult, Freq
from tno.quantum.utils.serialization import check_serializable

if TYPE_CHECKING:
    from pathlib import Path


EXPECTED_VARIABLES = np.array([0, 1, 0], dtype=np.uint8)
EXPECTED_VALUE = -50


@pytest.fixture(scope="module", name="path")
def path_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_folder")


class TestWithBaseResult:
    """Test the ``ResultInterface`` with BaseResults."""

    @pytest.fixture(
        name="base_result",
        params=["010", [0, 1, 0], np.array([0, 1, 0])],
        ids=["str", "list", "array"],
    )
    def base_result_fixture(
        self, request: pytest.FixtureRequest
    ) -> tuple[str | list[int] | NDArray[np.int_], int]:
        """Create the same QUBO object from different sources."""
        return (request.param, -50)

    def test_init(
        self,
        base_result: tuple[str | list[int] | NDArray[np.int_], int],
    ) -> None:
        basic_result = BasicResult.from_result(*base_result)
        np.testing.assert_array_equal(basic_result.best_bitvector, EXPECTED_VARIABLES)
        assert basic_result.best_value == EXPECTED_VALUE

    def test_serialization(
        self,
        base_result: tuple[str | list[int] | NDArray[np.int_], int],
    ) -> None:
        basic_result = BasicResult.from_result(*base_result)
        basic_result.execution_time = timedelta(10)
        basic_result.num_attempts = 1

        check_serializable(basic_result)


@pytest.fixture(scope="class", name="small_basicresult")
def small_basicresult_fixture() -> BasicResult:
    return BasicResult.from_result("010", -50)


def test_check_linear_equality_constraint() -> None:
    # construct a one-hot constraint for 4 variables with 3 values
    constraint_matrix = np.zeros((4, 12))
    for i in range(4):
        constraint_matrix[i, i * 3 : (i + 1) * 3] = 1
    constraint_vector = np.full(4, 1)

    sro = BasicResult.from_result("100010001010", -50)
    feasible = sro.check_linear_equality_constraint(
        constraint_matrix, constraint_vector
    )
    np.testing.assert_array_equal(feasible, True)

    sro = BasicResult.from_result("101010101010", -50)
    feasible = sro.check_linear_equality_constraint(
        constraint_matrix, constraint_vector
    )
    np.testing.assert_array_equal(feasible, [False, True, False, True])


def test_check_linear_inequality_constraint() -> None:
    # construct a one-hot constraint for 4 variables with 3 values
    constraint_matrix = [[1, 2, 3], [1, 1, 1]]
    constraint_vector = [3, 3]

    sro = BasicResult.from_result("110", -50)
    feasible = sro.check_linear_inequality_constraint(
        constraint_matrix, constraint_vector
    )
    np.testing.assert_array_equal(feasible, True)

    sro = BasicResult.from_result("111", -50)
    feasible = sro.check_linear_inequality_constraint(
        constraint_matrix, constraint_vector
    )
    np.testing.assert_array_equal(feasible, [False, True])


def test_get_energy_quantiles(small_basicresult: BasicResult) -> None:
    freq = Freq(["00", "01", "10", "11"], [-50, -4, 10, 12], [10, 5, 3, 2])
    small_basicresult._freq = freq
    q_computed = small_basicresult.get_energy_quantiles()
    np.testing.assert_array_equal(q_computed, [-50, -50, -27, -0.5, 12])
    assert small_basicresult.get_energy_quantiles(0) == freq.energies[0]
    assert small_basicresult.get_energy_quantiles(q=1) == freq.energies[3]


class TestHistPlot:
    """Tests for the hist_plot method of the SolverResultObject."""

    @pytest.fixture(scope="class")
    def small_basicresult(self, small_basicresult: BasicResult) -> BasicResult:
        small_basicresult._freq = Freq(["0", "1"], [-50, 0], [9, 1])
        return small_basicresult

    def test_expected_return(self, small_basicresult: BasicResult) -> None:
        # Check data mode
        data = small_basicresult.get_hist_bin_data(num_bins=4)
        np.testing.assert_array_equal(data[0], [9, 0, 0, 1])
        np.testing.assert_array_equal(data[1], [-50, -37.5, -25, -12.5, 0])

    def test_plot(self, small_basicresult: BasicResult) -> None:
        # Check if plot mode doesn't crash
        small_basicresult.plot_hist()

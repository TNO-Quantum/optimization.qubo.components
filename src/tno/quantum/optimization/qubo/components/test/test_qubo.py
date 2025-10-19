"""This module contains tests for the QUBO class."""

from __future__ import annotations

import itertools
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from dimod import BinaryQuadraticModel
from numpy.typing import NDArray
from tno.quantum.utils.serialization import Serializable

from tno.quantum.optimization.qubo.components import QUBO

if TYPE_CHECKING:
    from tno.quantum.utils import BitVectorLike


# Test Qubo: [[1, 2, 3], [4, -50, 6], [7, 8, 9]]
QUBO_MATRIX = [[1, 2, 3], [4, -50, 6], [7, 8, 9]]
QUBO_DICT = {
    (0, 0): 1,
    (0, 1): 2,
    (0, 2): 3,
    (1, 0): 4,
    (1, 1): -50,
    (1, 2): 6,
    (2, 0): 7,
    (2, 1): 8,
    (2, 2): 9,
}
expected_solution: dict[str, float] = {
    "000": 0,
    "001": 9,
    "010": -50,
    "011": -27,
    "100": 1,
    "101": 20,
    "110": -43,
    "111": -10,
}

RNG = np.random.default_rng()


@pytest.fixture(
    name="qubo",
    params=[
        QUBO_MATRIX,
        np.array(QUBO_MATRIX),
        QUBO_DICT,
        BinaryQuadraticModel.from_qubo(QUBO_MATRIX),  # type: ignore[arg-type]
    ],
    ids=["list", "array", "dict", "bqm"],
)
def qubo_fixture(request: pytest.FixtureRequest) -> QUBO:
    """Create the same QUBO object from different sources."""
    return QUBO(request.param)


# region serializable


def test_qubo_serializable(qubo: QUBO) -> None:
    """Test serializable.

    We can't use `check_serializable` due to rounding errors with floats.
    """
    upper_tri_qubo = qubo.to_upper_tri_form(copy=True)

    assert isinstance(qubo, Serializable), "Object is not Serializable"

    restore_from_json: QUBO = Serializable.from_json(qubo.to_json())
    np.testing.assert_almost_equal(restore_from_json.matrix, upper_tri_qubo.matrix)

    restore_from_yaml: QUBO = Serializable.from_yaml(qubo.to_yaml())
    np.testing.assert_almost_equal(restore_from_yaml.matrix, upper_tri_qubo.matrix)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test to and from JSON file
        temp_file_path = Path(temp_dir) / "test_file.json"
        qubo.to_json_file(temp_file_path)
        restore_from_json_file: QUBO = Serializable.from_json_file(temp_file_path)
        np.testing.assert_almost_equal(
            restore_from_json_file.matrix, upper_tri_qubo.matrix
        )

        temp_file_path = Path(temp_dir) / "test_file.yaml"
        qubo.to_yaml_file(temp_file_path)
        restore_from_yaml_file: QUBO = Serializable.from_yaml_file(temp_file_path)
        np.testing.assert_almost_equal(
            restore_from_yaml_file.matrix, upper_tri_qubo.matrix
        )


# region evaluate qubo


def test_evaluate_single(qubo: QUBO) -> None:
    """Test the evaluate method for single bitstrings."""
    for bit_str, expected_value in expected_solution.items():
        assert qubo.evaluate(bit_str) == expected_value

        bit_arr = np.array([int(el) for el in bit_str])
        assert qubo.evaluate(bit_arr) == expected_value


def test_evaluate_offset() -> None:
    """Test the evaluate method takes into account the QUBO offset."""
    qubo = QUBO([[0]], offset=123.0)
    assert qubo.evaluate("0") == qubo.offset


def test_evaluate_dict(qubo: QUBO) -> None:
    """Test the evaluate method for a mapping of bitstrings."""
    freq_dict: Mapping[BitVectorLike, float] = {"000": 0.5, "101": 0.2, "111": 0.3}

    true_value = 0.0
    for bit_str, freq in freq_dict.items():
        true_value += freq * qubo.evaluate(bit_str)

    assert qubo.evaluate_weighted(freq_dict) == true_value


# region convert qubo format


def test_to_upper_tri(qubo: QUBO) -> None:
    """Test the to_upper_tri_form method of the QUBO class."""
    output = qubo.to_upper_tri_form()
    np.testing.assert_array_equal(qubo.matrix, [[1, 6, 10], [0, -50, 14], [0, 0, 9]])
    assert output is qubo, "to_upper_tri_form did not return the QUBO object"


def test_to_upper_tri_copy(qubo: QUBO) -> None:
    """Test the to_upper_tri_form method of the QUBO class with copy flag."""
    qubo_matrix_copy = qubo.matrix.copy()
    qubo_copy = qubo.to_upper_tri_form(copy=True)
    np.testing.assert_array_equal(
        qubo_copy.matrix, [[1, 6, 10], [0, -50, 14], [0, 0, 9]]
    )
    np.testing.assert_array_equal(qubo_matrix_copy, qubo.matrix)


def test_to_symmetric_form(qubo: QUBO) -> None:
    """Test the to_symmetric_form method of the QUBO class."""
    output = qubo.to_symmetric_form()
    np.testing.assert_array_equal(qubo.matrix, [[1, 3, 5], [3, -50, 7], [5, 7, 9]])
    assert output is qubo, "to_symmetric_form did not return the QUBO object"


def test_to_symmetric_form_copy(qubo: QUBO) -> None:
    """Test the to_symmetric_form method of the QUBO class with copy flag."""
    qubo_matrix_copy = qubo.matrix.copy()
    qubo_copy = qubo.to_symmetric_form(copy=True)
    np.testing.assert_array_equal(qubo_copy.matrix, [[1, 3, 5], [3, -50, 7], [5, 7, 9]])
    np.testing.assert_array_equal(qubo_matrix_copy, qubo.matrix)


def test_to_ising(qubo: QUBO) -> None:
    """Test if the transformation QUBO->bqm when to labels are given."""
    external_fields, interactions, offset = qubo.to_ising()

    expected_external_fields = (-4.5, 20, -10.5)
    expected_interactions = ((0, 1.5, 2.5), (0, 0, 3.5), (0, 0, 0))
    expected_offset = -12.5

    np.testing.assert_array_equal(external_fields, expected_external_fields)
    np.testing.assert_array_equal(interactions, expected_interactions)
    assert offset == expected_offset


def test_to_and_from_bqm() -> None:
    """Test if the transformation bqm->QUBO->bqm gives the same bqm."""
    bqm1 = BinaryQuadraticModel({"x": 1, "y": 2, "z": 3}, {("x", "y"): 4}, 5, "BINARY")
    qubo, variables = QUBO.from_bqm(bqm1)
    bqm2 = qubo.to_bqm(variables)
    assert bqm1 == bqm2


def test_sort_labels() -> None:
    bqm = BinaryQuadraticModel("BINARY")
    bqm.add_linear("x", 1)
    bqm.add_linear("z", 3)
    bqm.add_linear("y", 2)
    qubo_unsorted, variables_unsorted = QUBO.from_bqm(bqm, sort_labels=False)
    qubo_sorted, variables_sorted = QUBO.from_bqm(bqm, sort_labels=True)

    assert variables_unsorted == ["x", "z", "y"]
    np.testing.assert_array_equal(
        qubo_unsorted.matrix, [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
    )

    assert variables_sorted == ["x", "y", "z"]
    np.testing.assert_array_equal(qubo_sorted.matrix, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])


def test_to_bqm_no_variables(qubo: QUBO) -> None:
    """Test the transformation QUBO->bqm when to labels are given."""
    bqm = BinaryQuadraticModel(
        {0: 1, 1: -50, 2: 9}, {(0, 1): 6, (0, 2): 10, (1, 2): 14}, 0, "BINARY"
    )
    assert bqm == qubo.to_bqm()


@pytest.mark.parametrize("qubo", [QUBO(QUBO_MATRIX)])
def test_negate_qubo(qubo: QUBO) -> None:
    """Test the negate_qubo method of the QUBO class."""
    output = qubo.negate()
    np.testing.assert_array_equal(
        qubo.matrix, [[-1, -2, -3], [-4, 50, -6], [-7, -8, -9]]
    )
    assert output is qubo, "negate_qubo did not return the QUBO object"


@pytest.mark.parametrize("qubo", [QUBO(QUBO_MATRIX)])
def test_negate_qubo_copy(qubo: QUBO) -> None:
    """Test the negate_qubo method of the QUBO class with copy flag."""
    qubo_matrix_copy = qubo.matrix.copy()
    qubo_copy = qubo.negate(copy=True)
    np.testing.assert_array_equal(
        qubo_copy.matrix, [[-1, -2, -3], [-4, 50, -6], [-7, -8, -9]]
    )
    np.testing.assert_array_equal(qubo.matrix, qubo_matrix_copy)
    assert qubo_copy is not qubo


# region arithmetic operations


def test_is_equal(qubo: QUBO) -> None:
    """Test the __eq__ method of the QUBO class."""
    assert qubo == QUBO(QUBO_MATRIX)
    assert qubo != QUBO(QUBO_MATRIX, offset=1)
    assert qubo != QUBO(np.zeros((3, 3)))


def test_addition() -> None:
    """Test the __add__ method of the QUBO class."""
    qubo_matrix1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    qubo_matrix2 = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
    qubo_matrix3 = [[2, 4, 6], [3, 5, 7], [6, 8, 10]]

    qubo1 = QUBO(qubo_matrix1)
    qubo2 = QUBO(qubo_matrix2)
    qubo3 = QUBO(qubo_matrix3)
    assert qubo3 == qubo1 + qubo2

    qubo3 = QUBO(qubo_matrix3, 5)
    assert qubo3 != qubo1 + qubo2

    qubo1 = QUBO(qubo_matrix1, 5)
    assert qubo3 == qubo1 + qubo2

    with pytest.raises(TypeError):
        qubo1 + "test"


def test_subtraction() -> None:
    """Test the __sub__ method of the QUBO class."""
    qubo_matrix1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    qubo_matrix2 = [[1, 2, 3], [2, 3, 4], [5, 6, 7]]
    qubo_matrix3 = [[0, 0, 0], [-1, -1, -1], [-4, -4, -4]]

    qubo1 = QUBO(qubo_matrix1)
    qubo2 = QUBO(qubo_matrix2)
    qubo3 = QUBO(qubo_matrix3)
    assert qubo3 == qubo1 - qubo2

    qubo3 = QUBO(qubo_matrix3, -5)
    assert qubo3 != qubo1 - qubo2

    qubo2 = QUBO(qubo_matrix2, 5)
    assert qubo3 == qubo1 - qubo2

    with pytest.raises(TypeError):
        qubo1 + "test"


def test_multiplication() -> None:
    """Test the __mul__ method of the QUBO class."""
    qubo_matrix1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    qubo_matrix2 = [[5, 10, 15], [5, 10, 15], [5, 10, 15]]

    qubo1 = QUBO(qubo_matrix1)
    qubo2 = QUBO(qubo_matrix2)
    assert qubo2 == qubo1 * 5
    assert qubo2 == 5 * qubo1

    qubo1 = QUBO(qubo_matrix1, offset=1)
    qubo2 = QUBO(qubo_matrix2, offset=5)
    assert qubo2 == qubo1 * 5
    assert qubo2 == 5 * qubo1

    qubo1 = QUBO(qubo_matrix1, offset=1)
    qubo2 = QUBO(qubo_matrix2, offset=1)
    assert qubo2 != qubo1 * 5
    assert qubo2 != 5 * qubo1


def test_division() -> None:
    """Test the __div__ method of the QUBO class."""
    qubo_matrix1 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    qubo_matrix2 = qubo_matrix1 / 5

    qubo1 = QUBO(qubo_matrix1)
    qubo2 = QUBO(qubo_matrix2)
    assert qubo2 == qubo1 / 5

    qubo2 = QUBO(qubo_matrix2, 1)
    assert qubo2 != qubo1 / 5

    qubo1 = QUBO(qubo_matrix1, 5)
    assert qubo2 == qubo1 / 5


def test_len(qubo: QUBO) -> None:
    """Test the __len__ method of the QUBO class."""
    qubo2_size = 100
    qubo2 = QUBO(np.zeros(shape=[qubo2_size, qubo2_size]))
    assert len(qubo) == len(QUBO_MATRIX)
    assert len(qubo2) == qubo2_size
    assert len(qubo) != len(qubo2)


# region delta x

permutations = list(itertools.product([0, 1], repeat=3))
correct_deltas = [
    np.array([1, -50, 9]),
    np.array([11, -36, -9]),
    np.array([7, 50, 23]),
    np.array([17, 36, -23]),
    np.array([-1, -44, 19]),
    np.array([-11, -30, -19]),
    np.array([-7, 44, 33]),
    np.array([-17, 30, -33]),
]


@pytest.mark.parametrize(
    ("x", "correct_delta"), zip(permutations, correct_deltas, strict=False)
)
def test_delta_x(x: NDArray[np.int_], correct_delta: NDArray[np.float64]) -> None:
    """Test if the delta_x function returns a correct derivative"""
    qubo = QUBO(QUBO_MATRIX)
    delta_x = qubo.delta_x(x)
    np.testing.assert_array_equal(delta_x, correct_delta)


permutations = list(itertools.product([0, 1], repeat=3))
correct_deltas = [
    np.array([[1.0, -43.0, 20.0], [0.0, -50.0, -27.0], [0.0, 0.0, 9.0]]),
    np.array([[11.0, -19.0, -8.0], [0.0, -36.0, -59.0], [0.0, 0.0, -9.0]]),
    np.array([[7.0, 51.0, 40.0], [0.0, 50.0, 59.0], [0.0, 0.0, 23.0]]),
    np.array([[17.0, 47.0, -16.0], [0.0, 36.0, 27.0], [0.0, 0.0, -23.0]]),
    np.array([[-1.0, -51.0, 8.0], [0.0, -44.0, -11.0], [0.0, 0.0, 19.0]]),
    np.array([[-11.0, -47.0, -20.0], [0.0, -30.0, -63.0], [0.0, 0.0, -19.0]]),
    np.array([[-7.0, 43.0, 16.0], [0.0, 44.0, 63.0], [0.0, 0.0, 33.0]]),
    np.array([[-17.0, 19.0, -40.0], [0.0, 30.0, 11.0], [0.0, 0.0, -33.0]]),
]


@pytest.mark.parametrize(
    ("x", "correct_delta"), zip(permutations, correct_deltas, strict=False)
)
def test_delta_x2(x: NDArray[np.int_], correct_delta: NDArray[np.float64]) -> None:
    """Test if the delta_x function returns a correct derivative"""
    qubo = QUBO(QUBO_MATRIX)
    delta_x2 = qubo.delta_x2(x)
    np.testing.assert_array_equal(delta_x2, correct_delta)


# region eigenvalues


@pytest.mark.parametrize(
    "qubo",
    [
        QUBO(np.array([[1, 2], [2, 1]])),
        QUBO(np.array([[4, -2], [-2, 4]])),
        QUBO(np.array([[0, 1], [1, 0]])),
        QUBO(QUBO_MATRIX),
        QUBO(RNG.random(size=(5, 5))),
    ],
)
def test_eig(qubo: QUBO) -> None:
    """Test the eig() function of the QUBO class with multiple matrices."""
    eigenvalues, eigenvectors = qubo.eig()
    np_eigenvalues, np_eigenvectors = np.linalg.eig(qubo.matrix)

    # Type checks
    assert isinstance(eigenvalues, np.ndarray)
    assert isinstance(eigenvectors, np.ndarray)

    # Format checks
    assert eigenvalues.shape == np_eigenvalues.shape
    assert eigenvectors.shape == np_eigenvectors.shape

    # Correctness checks
    np.testing.assert_array_almost_equal(eigenvalues, np_eigenvalues)
    np.testing.assert_array_almost_equal(np.abs(eigenvectors), np.abs(np_eigenvectors))


# region plots


def test_spectral_gap() -> None:
    """Test spectral_gap() computes the correct minimum spectral gap."""
    qubo = QUBO([[2, 1], [3, 4]])
    min_gap, _ = qubo.spectral_gap([1.0, 0.5, 0.0], [0.0, 0.5, 1.0], plot=False)
    assert min_gap == pytest.approx(0.724, 0.001)


def test_spectral_gap_raise_error() -> None:
    """Test spectral_gap() raises ValueError on invalid inputs."""
    qubo = QUBO(QUBO_MATRIX)

    expected_error_msg = r"A and B must have positive length\."
    with pytest.raises(ValueError, match=expected_error_msg):
        qubo.spectral_gap([], [], plot=False)

    expected_error_msg = r"A and B do not have equal length \(2 and 3\)\."
    with pytest.raises(ValueError, match=expected_error_msg):
        qubo.spectral_gap([1.0, 0.0], [0.0, 0.5, 1.0], plot=False)


@pytest.mark.parametrize(
    ("qubo", "opt"),
    [
        (QUBO([[4, -1, -10], [17, 10, -15], [-5, 6, -13]]), -24.0),
        (QUBO([[9, 1, -8], [-4, -12, 8], [-19, 6, -20]]), -39.0),
        (QUBO([[-13, 12, 14], [11, -18, -17], [-5, 13, 11]]), -18.0),
        (QUBO([[-3, 17, 10], [19, -12, 1], [-17, 12, 17]]), -12.0),
        (QUBO([[19, -12, 17], [1, -7, -3], [-14, -20, -8]]), -38.0),
        (QUBO([[0]], +1000.0), +1000.0),
        (QUBO([[0]], -1000.0), -1000.0),
        (QUBO(np.zeros((0, 0)), 42.0), 42.0),
    ],
)
def test_ub_and_lb(qubo: QUBO, opt: float) -> None:
    """Test the upper and lower bounds with multiple matrices."""
    assert qubo.lower_bound <= opt
    assert opt <= qubo.upper_bound


def test_compute_bounds_raise_warning() -> None:
    """Test the warning of compute_bounds."""
    qubo = QUBO([[4, -1, -10], [17, 10, -15], [-5, 6, -13]])
    expected_message = (
        "QUBO bounds are being recomputed but no optimization arguments were passed"
    )
    qubo.compute_bounds()
    with pytest.warns(UserWarning, match=re.escape(expected_message)):
        qubo.compute_bounds()

"""Tests for the ``PartialSolution`` class."""

from __future__ import annotations

import pytest

from tno.quantum.optimization.qubo.components import QUBO, PartialSolution
from tno.quantum.utils import BitVector
from tno.quantum.utils.serialization import check_serializable


def test_assign_variable() -> None:
    """Test if variables are correctly assigned to other variables and resolved.

    Given the assignments
        x_0 := x_1
        x_1 := x_2
        x_3 := 1 - x_4
        x_4 := x_5
    we expect to find that
        x_0 = x_2
        x_1 = x_2
        x_2 = x_2
        x_3 = 1 - x_5
        x_4 = x_5
        x_5 = x_5
    """
    p = PartialSolution(6)
    p.assign_variable(0, 1)
    p.assign_variable(1, 2)
    p.assign_variable(3, 4, conj=True)
    p.assign_variable(4, 5, conj=False)
    assert p._resolve(0) == (2, False)
    assert p._resolve(1) == (2, False)
    assert p._resolve(2) == (2, False)
    assert p._resolve(3) == (5, True)
    assert p._resolve(4) == (5, False)
    assert p._resolve(5) == (5, False)


def test_assign_value() -> None:
    """Test if variables are correctly assigned to values and resolved.

    Given the assignments
        x_0 := 1 - x_1
        x_1 := 0
    we expect to find that
        x_0 = 1
        x_1 = 0
    """
    p = PartialSolution(2)
    p.assign_variable(0, 1, conj=True)
    p.assign_value(1, 0)
    assert p._resolve(0) == (-1, True)
    assert p._resolve(1) == (-1, False)


@pytest.mark.parametrize(("i", "j"), [(-1, 0), (3, 0), (0, -1), (0, 3)])
def test_assign_variable_raise_error_bounds(i: int, j: int) -> None:
    """Test assign_variable() raise ValueError on out of bounds input"""
    p = PartialSolution(3)
    expected_error_msg = "bound"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.assign_variable(i, j)


@pytest.mark.parametrize("i", [-1, 3])
def test_assign_value_raise_error_bounds(i: int) -> None:
    """Test assign_value() raise ValueError on out of bounds input"""
    p = PartialSolution(3)
    expected_error_msg = "bound"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.assign_value(i, 0)


def test_assign_variable_raise_error_assigned() -> None:
    """Test assign_variable() raises ValuError on already assigned variable"""
    p = PartialSolution(3)
    p.assign_variable(0, 1)
    expected_error_msg = "already assigned"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.assign_variable(0, 2)


def test_assign_value_raise_error_assigned() -> None:
    """Test assign_value() raises ValuError on already assigned variable"""
    p = PartialSolution(3)
    p.assign_variable(0, 1)
    expected_error_msg = "already assigned"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.assign_value(0, 1)


def test_assign_variable_raise_error_cycle() -> None:
    """Test assign_value() raises ValuError on assignment cycle"""
    p = PartialSolution(3)
    p.assign_variable(0, 1)
    p.assign_variable(1, 2)
    expected_error_msg = "cycle"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.assign_variable(2, 0)


def test_expand() -> None:
    """Test expand() expands correctly.

    Example assignment:
        x_0 := 1 - x_1
        x_1 := 1 - x_2
        x_3 := 1 - x_4
        x_4 := 0

    Expecting `x_2 = 0` to expand in `x = [0, 1, 0, 1, 0]`.
    """
    n = 5
    p = PartialSolution(n)
    p.assign_variable(0, 1, conj=True)
    p.assign_variable(1, 2, conj=True)
    p.assign_variable(3, 4, conj=True)
    p.assign_value(4, 0)
    assert p.expand([0]) == BitVector("01010")


def test_expand_rasise_error() -> None:
    """Test expand() raises ValueError on invalid input"""
    n = 5
    p = PartialSolution(n)
    expected_error_msg = "length"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.expand([0, 1])


def test_apply() -> None:
    """Test apply() converts QUBO correctly.

    Example QUBO:
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]

    From assignment x_0 := x_1 and x_2 := 1, expect to obtain:
        [ 36 ]  ( with offset = 9 )
    """
    qubo = QUBO(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    p = PartialSolution(qubo.size)
    p.assign_variable(0, 1)
    p.assign_value(2, 1)

    qubo_reduced = p.apply(qubo)

    assert qubo_reduced == QUBO([[36]], 9)


def test_apply_raise_error() -> None:
    """Test apply() raises ValueError on invalid sized QUBO"""
    p = PartialSolution(3)
    expected_error_msg = "size"
    with pytest.raises(ValueError, match=expected_error_msg):
        p.apply(QUBO([[0]]))


def test_repr() -> None:
    """Test __repr__() yields expected representation."""
    p = PartialSolution(5)
    p.assign_variable(0, 1, conj=True)
    p.assign_variable(1, 2)
    p.assign_value(3, 0)
    p.assign_value(4, 1)

    assert repr(p) == "PartialSolution {x_0 = 1 - x_2, x_1 = x_2, x_3 = 0, x_4 = 1}"


def test_serialize() -> None:
    """Test serialize()"""
    partial_solution = PartialSolution(10)
    partial_solution.assign_value(0, 0)
    partial_solution.assign_value(1, 1)
    partial_solution.assign_variable(2, 3)
    partial_solution.assign_variable(4, 5, conj=True)

    assert partial_solution.serialize() == {
        "__class__": PartialSolution._class_to_path(PartialSolution),
        "n": 10,
        "x0": 0,
        "x1": 1,
        "x2": "x3",
        "x4": "1 - x5",
    }


def test_deserialize() -> None:
    """Test deserialize()"""
    partial_solution = PartialSolution.deserialize(
        {
            "__class__": PartialSolution._class_to_path(PartialSolution),
            "n": 10,
            "x0": 0,
            "x1": 1,
            "x2": "x3",
            "x4": "1 - x5",
        }
    )

    assert partial_solution._n == 10
    assert partial_solution._resolve(0) == (-1, 0)
    assert partial_solution._resolve(1) == (-1, 1)
    assert partial_solution._resolve(2) == (3, 0)
    assert partial_solution._resolve(4) == (5, 1)


def test_serializable() -> None:
    """Test (de)serialization."""
    partial_solution = PartialSolution(10)
    partial_solution.assign_value(0, 0)
    partial_solution.assign_value(1, 1)
    partial_solution.assign_variable(2, 3)
    partial_solution.assign_variable(4, 5, conj=True)

    check_serializable(partial_solution)


def test_eq() -> None:
    """Test equality of PartialSolution objects."""
    p = PartialSolution(3)
    p.assign_variable(0, 1)
    p.assign_variable(1, 2)

    q = PartialSolution(3)
    q.assign_variable(0, 2)
    q.assign_variable(1, 2)

    assert p == q


def test_not_eq() -> None:
    """Test inequality of PartialSolution objects."""
    p = PartialSolution(3)
    p.assign_variable(0, 1)
    p.assign_variable(1, 2)

    q = PartialSolution(3)
    q.assign_variable(0, 2)
    q.assign_variable(1, 2, conj=True)

    assert p != q

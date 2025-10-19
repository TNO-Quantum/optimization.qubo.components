"""This module contains test for the ``Freq`` class."""

from __future__ import annotations

from collections.abc import Iterable
from typing import SupportsFloat, SupportsInt

import numpy as np
import pytest
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import check_serializable

from tno.quantum.optimization.qubo.components import Freq


class TestFreq:
    """Tests for the Freq object."""

    @pytest.fixture(name="freq")
    def freq_fixture(self) -> Freq:
        return Freq(["01"], [2], [3])

    @pytest.mark.parametrize(
        argnames=(
            "bitvectors",
            "energies",
            "num_occurrences",
            "expected_bitvectors",
            "expected_energies",
            "expected_num_occurrences",
        ),
        argvalues=[
            ([(0, 1)], [2.0], [3], [BitVector("01")], [2.0], [3]),
            (
                ["01", "10"],
                [2.0, 3],
                [3, 1],
                [BitVector("01"), BitVector("10")],
                [2.0, 3.0],
                [3, 1],
            ),
            (
                [[0, 1], [1, 0]],
                (2.0, 3),
                (3, 1),
                [BitVector("01"), BitVector("10")],
                [2.0, 3.0],
                [3, 1],
            ),
            (
                [np.array([0, 1]), np.array([1, 0])],
                [2.0, 3],
                [3, 1],
                [BitVector("01"), BitVector("10")],
                [2.0, 3.0],
                [3, 1],
            ),
        ],
    )
    def test_attributes(  # noqa: PLR0913
        self,
        bitvectors: Iterable[BitVectorLike],
        energies: Iterable[SupportsFloat],
        num_occurrences: Iterable[SupportsInt],
        expected_bitvectors: list[BitVector],
        expected_energies: list[float],
        expected_num_occurrences: list[int],
    ) -> None:
        """Test attributes are set correctly."""
        freq = Freq(bitvectors, energies, num_occurrences)
        assert freq.bitvectors == expected_bitvectors
        assert freq.energies == expected_energies
        assert freq.num_occurrences == expected_num_occurrences

    def test_serializable(self) -> None:
        """Test (de)serializatoin."""
        freq = Freq(["01", "10"], [2.0, 3.0], [4, 5])
        check_serializable(freq)

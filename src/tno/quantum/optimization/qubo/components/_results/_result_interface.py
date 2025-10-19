"""This module contains the abstract ``ResultInterface`` class.

This class can be used to store the results returned from the solve method, and includes
functions for storing, loading, visualizing and other helper functions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsInt

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import Serializable
from tno.quantum.utils.validation import check_int, check_real, check_timedelta

if TYPE_CHECKING:
    from typing import Self

    from matplotlib.axes import Axes

    from tno.quantum.optimization.qubo.components._freq import Freq

logger = logging.getLogger(__name__)

DEFAULT_PLOT_NBINS = 20


plot_settings: dict[str, Any]
plot_settings = {
    "color": "#649EC9",
    "fontsize": {"title": 24, "axes": 20, "legend": 20, "labelsize": 16},
    "ticksize": (3, 12),
    "spine_linewidth": 2,
    "figsize": (12, 9),
}


class ResultInterface(ABC, Serializable):
    """Abstract result base class representing the result of a solver."""

    def __init__(
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        *,
        execution_time: timedelta | SupportsInt = 0,
        num_attempts: SupportsInt = 1,
    ) -> None:
        """Init of :py:class:`ResultInterface`.

        Args:
            best_bitvector: Bitvector corresponding to the best result.
            best_value: Objective value corresponding to the best result.
            freq: Frequency object containing the frequency of found bitvectors and
                energies.
            execution_time: Time to successfully execute the solve method.
            num_attempts: Number of attempts it took to successfully execute the solve
                method.
        """
        self._best_bitvector = BitVector(best_bitvector)
        self._best_value = check_real(best_value, "best_value")
        self._freq = freq

        self._execution_time = check_timedelta(
            execution_time, "execution_time", l_bound=0
        )
        self._num_attempts = check_int(num_attempts, "num_attempts", l_bound=0)

    @classmethod
    @abstractmethod
    def from_result(cls, *args: Any, **kwargs: Any) -> Self:
        """Abstract method which parses the result of solver backend.

        The arguments for this function are determined by the implementing class.
        """

    @property
    def best_bitvector(self) -> BitVector:
        """Bitvector corresponding to the best result."""
        return self._best_bitvector

    @property
    def best_value(self) -> float:
        """Objective value corresponding to the best result."""
        return self._best_value

    @property
    def freq(self) -> Freq:
        """Frequency object containing frequency of found bitvectors and energies."""
        return self._freq

    @property
    def execution_time(self) -> timedelta:
        """Time to successfully execute the solve method."""
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: Any) -> None:
        self._execution_time = check_timedelta(value, "execution_time", l_bound=0)

    @property
    def num_attempts(self) -> int:
        """Number of attempts it took to successfully execute the solve method."""
        return self._num_attempts

    @num_attempts.setter
    def num_attempts(self, value: Any) -> None:
        self._num_attempts = check_int(value, "num_attempts", l_bound=0)

    def get_energy_quantiles(
        self,
        q: ArrayLike = [0, 0.25, 0.5, 0.75, 1.0],  # noqa: B006
        **kwargs: Any,
    ) -> float | NDArray[np.float64]:
        """Computes the quantiles of the energy.

        Args:
            q: Quantile or sequence of quantiles to compute. Each values of `q` must be
                in the interval $[0, 1]$ . Defaults to ``[0, 0.25, 0.5, 0.75, 1.0]``.
            kwargs: Additional keyword arguments for the ``numpy.quantile`` method.

        Returns:
            If `q` is a single quantile, then the result is a scalar representing the
            quantile `q` of the energy. If multiple quantiles are given, then an
            :py:class:`NDArray` is returned with the requested quantile values.
        """
        temp = []
        for _, energy, occurrences in self.freq:
            temp += [energy] * occurrences
        if isinstance(q, float):
            return float(np.quantile(temp, q, **kwargs))

        return np.asarray(np.quantile(temp, np.asarray(q), **kwargs), dtype=np.float64)

    def plot_hist(
        self,
        num_bins: int = DEFAULT_PLOT_NBINS,
        ax: Axes | None = None,
    ) -> None:
        """Plot a histogram of the result.

        Args:
            num_bins: Number of bins of the histogram. Defaults to 20.
            ax: Axes to plot on. If ``None`` (default) a new figure with axis is
                created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=plot_settings["figsize"])

        ax.hist(
            x=self.freq.energies,
            bins=num_bins,
            weights=self.freq.num_occurrences,
            color=plot_settings["color"],
        )

        ax.set_title("Energy Histogram", fontsize=plot_settings["fontsize"]["title"])
        ax.set_xlabel("Energies", fontsize=plot_settings["fontsize"]["axes"])
        ax.set_ylabel(
            "Number of Occurrences", fontsize=plot_settings["fontsize"]["axes"]
        )
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.tick_params(
            labelsize=plot_settings["fontsize"]["labelsize"],
            width=plot_settings["ticksize"][0],
            length=plot_settings["ticksize"][1],
        )

    def get_hist_bin_data(
        self, num_bins: int = DEFAULT_PLOT_NBINS
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Computes histogram of frequency energies and return the bins and their sizes.

        Args:
            num_bins: Number of bins of the histogram. Defaults to 20.

        Returns:
                Two :py:class:`NDArray` s. The first array is the height of each bin
                and the second gives the edges of the bins.
        """
        return np.histogram(
            self.freq.energies, bins=num_bins, weights=self.freq.num_occurrences
        )

    def check_linear_equality_constraint(
        self,
        A: ArrayLike,  # noqa: N803
        b: ArrayLike,
    ) -> NDArray[np.bool_]:
        """Checks if the linear equality constraint $Ax = b$ is met.

        Args:
            A: Coefficient matrix of the constraint.
            b: Right hand side vector.

        Returns:
            :py:class:`NDArray` with boolean values for each row of $A$. Element $i$ is
            ``True`` if constraint $i$ is met ($A_i x = b_i$) and is ``False``
            otherwise.
        """
        return np.asarray(
            np.asarray(A) @ self.best_bitvector == np.asarray(b), dtype=bool
        )

    def check_linear_inequality_constraint(
        self,
        A: ArrayLike,  # noqa: N803
        b: ArrayLike,
    ) -> NDArray[np.bool_]:
        r"""Checks if the linear inequality constraint $Ax \le b$ is met.

        Args:
            A: Coefficient matrix of the constraint.
            b: Right hand side vector.

        Returns:
            :py:class:`NDArray` with boolean values for each row of $A$. Element $i$ is
            ``True`` if constraint $i$ is met ($A_ix \le b_i$) and is ``False``
            otherwise.
        """
        return np.asarray(
            np.asarray(A) @ self.best_bitvector <= np.asarray(b), dtype=bool
        )

    def __repr__(self) -> str:
        """String representation of result."""
        # Find public and @property attributes
        attributes = [
            "best_bitvector",
            "best_value",
            "execution_time",
            "num_attempts",
            "freq",
        ]
        for name in vars(self):  # public attributes
            if not name.startswith("_") and name not in attributes:
                attributes.append(name)
        for name in dir(self):  # @property attributes
            if not name.startswith("_") and name not in attributes:
                attr = getattr(type(self), name, None)
                if isinstance(attr, property):
                    attributes.append(name)

        return (
            f"{self.__class__.__name__}("
            f"best_bitvector={self.best_bitvector}, "
            f"best_value={self.best_value}, "
            f"execution_time={self.execution_time}, "
            f"num_attempts={self.num_attempts}, "
            f"freq=(Freq instance), "
            + ", ".join(
                [
                    f"{name}=({type(getattr(self, name)).__name__} instance)"
                    for name in attributes[5:]
                ]
            )
            + ")"
        )

    def _repr_html_(self) -> str:
        """HTML representation of result."""
        # Find public and @property attributes
        attributes = ["best_bitvector", "best_value", "execution_time", "num_attempts"]
        for name in vars(self):  # public attributes
            if not name.startswith("_") and name not in attributes:
                attributes.append(name)
        for name in dir(self):  # @property attributes
            if not name.startswith("_") and name not in attributes:
                attr = getattr(type(self), name, None)
                if isinstance(attr, property):
                    attributes.append(name)

        html = (
            "<style>"
            ".result-interface {"
            "   display: inline-block;"
            "   border: 1px solid black;"
            "   border-radius: 2px;"
            "   background-color: white;"
            "   color: black;"
            "   table { margin: 4px 0px; width: 100%; }"
            "   summary { padding: 4px 8px; }"
            "   td, th { text-align: left; padding: 4px 8px; }"
            "   th { font-weight: bold; }"
            "   .tt { font-family: monospace; }"
            "   .header { background-color: #002484; color: white; }"
            " }"
            "</style>"
        )
        html += "<div class='result-interface'>"
        html += "<details>"
        html += f"<summary class='header'>{self.__class__.__name__}</summary>"
        html += "<table>"

        for name in attributes:
            if name == "freq":  # special case, handled at the end
                continue
            html += f"<tr><td class='tt'>{name}</td>"
            value = getattr(self, name)
            if isinstance(value, (int, float, timedelta)):
                html += f"<td>{value}</td></tr>"
            elif isinstance(value, BitVector):
                html += f"<td class='tt'>{value}</td></tr>"
            else:
                html += (
                    "<td style='color: #777;'>("
                    f"<span class='tt'>{type(value).__name__}</span>"
                    " instance)</td>"
                )

        html += "</table>"
        html += "<details>"
        html += "<summary>Frequency data</summary>"
        html += "<table>"
        html += "<tr><th>Bitvector</th><th>Value</th><th>Occurrences</th></tr>"
        html += "".join(
            [
                f"<tr><td class='tt'>{bitvector}</td>"
                f"<td>{value}</td><td>{num_occurrence}</td></tr>"
                for bitvector, value, num_occurrence in self.freq
            ]
        )
        html += "</table>"
        html += "</details>"
        html += "</details>"
        html += "</div>"
        return html

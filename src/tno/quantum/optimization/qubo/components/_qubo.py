"""This module contains the ``QUBO`` class."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import Serializable

if importlib.util.find_spec("dimod") is not None:
    from dimod import BinaryQuadraticModel

if TYPE_CHECKING:
    from typing import Self

    from matplotlib.figure import Figure


class QUBO(Serializable):  # noqa: PLW1641
    """Class representing a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> QUBO([
        ...     [1, 2, 3],
        ...     [4, 5, 6],
        ...     [7, 8, 9]
        ... ])
        QUBO of dim: 3x3, with 9 non zero elements.
    """

    def __init__(
        self,
        data: ArrayLike | Mapping[tuple[int, int], float] | BinaryQuadraticModel,
        /,
        offset: float = 0,
    ) -> None:
        """Init :py:class:`QUBO`.

        Args:
            data: Valid QUBO matrix, or mapping containing matrix coefficients,
                or BQM.
            offset: The offset of the QUBO (constant value).
        """
        self._parse_data(data)
        self._offset: float = self._offset + offset if self._offset else offset
        self.check_valid()
        self._histogram_base: list[float] = []
        self.is_negated = False
        self._upper_bound: float | None = None
        self._lower_bound: float | None = None

    def _parse_data(
        self,
        data: ArrayLike | Mapping[tuple[int, int], float] | BinaryQuadraticModel,
        /,
    ) -> None:
        """Parse data properties from data.

        Args:
            data: Valid QUBO matrix or BQM.
        """
        # `BinaryQuadraticModel`
        if importlib.util.find_spec("dimod") is not None and isinstance(
            data, BinaryQuadraticModel
        ):
            qubo_object, _ = self.from_bqm(data)
            self._matrix = qubo_object.matrix
            self._offset = qubo_object.offset
            return

        # `Mapping[tuple[int, int], float]`
        if isinstance(data, Mapping):
            size = 1 + max(max(data, key=lambda x: max(x)))
            matrix = np.zeros((size, size), dtype=np.float64)
            for indices, value in data.items():
                matrix[indices] = value
            self._matrix = matrix.copy()
            self._offset = 0.0
            return

        # `ArrayLike`
        self._matrix = np.asarray(data).copy()
        self._offset = 0.0

    @classmethod
    def from_bqm(
        cls, bqm: BinaryQuadraticModel, *, sort_labels: bool = False
    ) -> tuple[Self, list[Any]]:
        """Construct a :py:class:`QUBO` instance from :py:class:`~dimod.binary.BinaryQuadraticModel`.

        This method converts a given :py:class:`~dimod.binary.BinaryQuadraticModel` into
        a :py:class:`QUBO` representation. If the
        :py:class:`~dimod.binary.BinaryQuadraticModel` variable type is `SPIN`, it will
        be automatically converted to `BINARY` with a warning. The resulting QUBO matrix
        is constructed using the linear and quadratic coefficients from the
        :py:class:`~dimod.binary.BinaryQuadraticModel`, and returned along with the list
        of variable labels used in the conversion. Optionally, the labels can be sorted
        as well.

        Args:
            bqm: The binary quadratic model to convert.
            sort_labels: If ``True``, variable labels will be sorted before constructing
                the QUBO matrix. Defaults to ``False``.

        Returns:
            Tuple containing a :py:class:`QUBO` instance representing the input
            :py:class:`~dimod.binary.BinaryQuadraticModel` and a (optionally sorted)
            list of variable labels corresponding to the order used in the QUBO matrix.

        Example:
            The example below shows how to generate a :py:class:`QUBO` object from a
            :py:class:`~dimod.binary.BinaryQuadraticModel`.

            >>> from dimod import BinaryQuadraticModel
            >>> bqm = BinaryQuadraticModel({"x": 1, "y": 2, "z": 3}, {("x", "y"): 4}, 5, "BINARY")
            >>> qubo, labels = QUBO.from_bqm(bqm)
            >>> labels
            ['x', 'y', 'z']
            >>> qubo.matrix
            array([[1., 0., 0.],
                   [4., 2., 0.],
                   [0., 0., 3.]])
            >>> qubo.offset
            5.0
        """  # noqa : E501
        if bqm.vartype.name == "SPIN":
            warnings.warn(
                "BQM has vartype 'SPIN', this will be changed to 'BINARY'", stacklevel=1
            )
            bqm = deepcopy(bqm)
            bqm.change_vartype("BINARY")  # type: ignore[arg-type]
        variables = sorted(bqm.variables) if sort_labels else list(bqm.variables)  # type: ignore[type-var]
        lin, (row, col, quad_val), offset = bqm.to_numpy_vectors(
            variable_order=variables
        )  # type: ignore[misc]
        qubo_matrix = np.zeros((len(variables), len(variables)))
        np.fill_diagonal(qubo_matrix, lin)
        qubo_matrix[row, col] = quad_val

        return cls(qubo_matrix, float(offset)), variables

    def to_bqm(self, variables: list[Any] | None = None) -> BinaryQuadraticModel:
        """Convert the :py:class:`QUBO` into a :py:class:`~dimod.binary.BinaryQuadraticModel`.

        Args:
            variables: list of variable labels. If ``None`` (default), integer labels
                will be used.

        Returns:
            The :py:class:`~dimod.binary.BinaryQuadraticModel` of the :py:class:`QUBO`
            with the variable labels described by `variables`.
        """  # noqa: E501
        bqm = BinaryQuadraticModel.from_qubo(self.matrix, self.offset)  # type: ignore[arg-type]
        if variables is not None:
            bqm.relabel_variables(dict(enumerate(variables)))
        return cast("BinaryQuadraticModel", bqm)

    def to_ising(self) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        r"""Convert the :py:class:`QUBO` into a Lenz-Ising model.

        The QUBO problem is transformed to the Lenz-Ising model problem by using the
        following change of variables: $x_i \to (1 - s_i) / 2$.

        Returns:
            Tuple of a 1D Array, 2D Array and float representing the external fields,
            interactions and offset of the corresponding Lenz-Ising model respectively.
        """
        external_fields = -0.25 * (self.matrix.sum(axis=0) + self.matrix.sum(axis=1))
        interactions = 0.25 * (np.triu(self.matrix, k=1) + np.tril(self.matrix, k=-1).T)
        offset = float(
            self.offset + self.matrix.diagonal().sum() / 2 + interactions.sum()
        )
        return external_fields, interactions, offset

    def check_valid(self) -> None:
        """Check the validity of the QUBO, if not valid raise an error.

        Raises:
            TypeError: If the `dtype` of the :py:class:`QUBO` matrix can not be cast to
                float.
            ValueError: If the :py:class:`QUBO` matrix is not square.
        """
        if not np.can_cast(self.matrix.dtype, float):
            error_msg = "Unrecognized dtype."
            raise TypeError(error_msg)

        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            error_msg = "The QUBO matrix is not square."
            raise ValueError(error_msg)

    def _serialize(self) -> dict[str, Any]:
        """Serialize QUBO to dict."""
        upper_tri_qubo = self.to_upper_tri_form(copy=True).matrix
        non_zero_entries = [
            [int(i), int(j), float(upper_tri_qubo[i, j])]
            for i, j in np.argwhere(upper_tri_qubo)
        ]
        return {
            "non_zero_entries": non_zero_entries,
            "offset": self.offset,
        }

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> QUBO:
        """Deserialize data into an instance of `QUBO`."""
        offset = data["offset"]
        matrix = {(i, j): value for (i, j, value) in data["non_zero_entries"]}
        return QUBO(matrix, offset=offset)

    def __eq__(self, other: object) -> bool:
        """Check if two QUBOs are equal.

        Two QUBOs are equal if they have equal symmetric form and have the same
        ``offset``.
        """
        if not isinstance(other, QUBO):
            return False

        if self.offset != other.offset:
            return False

        # Standardize qubo representation to symmetric form
        qubo1 = np.add(self.matrix, self.matrix.T) / 2
        qubo2 = np.add(other.matrix, other.matrix.T) / 2
        return np.array_equal(qubo1, qubo2)

    def __add__(self, rhs: object) -> QUBO:
        """Add two QUBOs element-wise."""
        if not isinstance(rhs, QUBO):
            error_msg = "Only addition between two instances of QUBO is supported."
            raise TypeError(error_msg)

        return QUBO(self.matrix + rhs.matrix, self.offset + rhs.offset)

    def __sub__(self, rhs: object) -> QUBO:
        """Subtract two QUBOs element-wise."""
        if not isinstance(rhs, QUBO):
            error_msg = "Only subtraction between two instances of QUBO is supported."
            raise TypeError(error_msg)

        return QUBO(self.matrix - rhs.matrix, self.offset - rhs.offset)

    def __mul__(self, scalar: float) -> QUBO:
        """Multiply the QUBO and offset by a scalar."""
        if not isinstance(scalar, Real):
            error_msg = (
                "Multiplication with a QUBO is only defined for real valued scalars."
            )
            raise TypeError(error_msg)

        return QUBO(self.matrix * scalar, self.offset * scalar)

    def __rmul__(self, scalar: float) -> QUBO:
        """Multiply the QUBO and offset by a scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> QUBO:
        """Division of the QUBO and offset by a scalar."""
        if not isinstance(scalar, Real):
            error_msg = "Division with a QUBO is only defined for real valued scalars."
            raise TypeError(error_msg)

        return QUBO(self.matrix / scalar, self.offset / scalar)

    def __repr__(self) -> str:
        """String representation of the QUBO."""
        non_zero = np.count_nonzero(self.matrix)
        return (
            f"QUBO of dim: {self.size}x{self.size}, with {non_zero} non zero elements."
        )

    def __len__(self) -> int:
        """Size of the QUBO, that is, its number of variables."""
        return self.size

    def to_symmetric_form(self, *, copy: bool = False) -> QUBO:
        """Rewrite QUBO into symmetric form.

        Args:
            copy: If true, the changes are applied to a copy of the QUBO, and this copy
                is returned.

        Returns:
            The rewritten QUBO, or a copy of it.
        """
        qubo = deepcopy(self) if copy else self

        if not np.array_equal(qubo.matrix, qubo.matrix.T):
            qubo._matrix = (qubo.matrix + qubo.matrix.T) / 2  # noqa: SLF001

        return qubo

    def to_upper_tri_form(self, *, copy: bool = False) -> QUBO:
        """Rewrite QUBO into upper triangular form.

        Args:
            copy: If true, the changes are applied to a copy of the QUBO, and this copy
                is returned.

        Returns:
            The rewritten QUBO, or a copy of it.
        """
        qubo = deepcopy(self) if copy else self

        if not np.array_equal(qubo.matrix, triu_qubo := np.triu(qubo.matrix)):
            qubo._matrix = triu_qubo + np.triu(qubo.matrix.T, k=1)  # noqa: SLF001

        return qubo

    def delta_x(self, x: BitVectorLike) -> NDArray[np.float64]:
        """Calculate the change in QUBO objective value due to a single bit flip.

        Effectively this computes the following:

        .. code-block:: python

            Q.delta_x(x)[i] = Q.evaluate(x.flip_indices(i)) - Q.evaluate(x)

        Args:
            x: Input bit vector.

        Returns:
            Change in QUBO objective value for all single-bit flips of the input vector.

        .. note::
            The computational complexity of this method is $O(n^2)$.
        """
        x = BitVector(x)
        # pre compute the interaction terms (complexity is O(n^2))
        diagonal = self.matrix.diagonal()
        interaction_terms = self.matrix @ x + self.matrix.T @ x
        signs = x.to_ising()

        # compute the n evaluations (complexity is O(n))
        return diagonal + signs * interaction_terms

    def delta_x2(self, x: BitVectorLike) -> NDArray[np.float64]:
        r"""Calculate the change in QUBO objective value due to two bit flips.

        Effectively this computes the following for `$j \ge i$`:

        .. code-block:: python

            Q.delta_x2(x)[i, j] = Q.evaluate(x.flip_indices(i, j)) - Q.evaluate(x)

        Args:
            x: Input bit vector.

        Returns:
            Changes in QUBO objective value for double-bit flips of the input vector.
            Values are returned in an upper triangular matrix.

        .. note::
            The computational complexity of this method is $O(n^2)$.
        """
        x = BitVector(x)

        # pre compute the interaction terms (complexity is O(n^2))
        interaction_terms = self.matrix @ x + self.matrix.T @ x
        signs = x.to_ising()

        # compute the n(n-1)/2 evaluations (complexity is O(n^2))
        # use vectorized numpy evaluation for speed
        evaluations = np.outer(
            self.matrix.diagonal() + signs * interaction_terms, np.ones(self.size)
        )
        evaluations += np.outer(signs, signs) * self.matrix
        evaluations = np.triu(evaluations + evaluations.T, k=1)
        np.fill_diagonal(evaluations, self.delta_x(x))

        return evaluations.astype(dtype=np.float64)

    def negate(self, *, copy: bool = False) -> QUBO:
        """Negate QUBO so that 'maximization' and 'minimization' are interchanged.

        Args:
            copy: If true, the changes are applied to a copy of the QUBO, and this copy
                is returned.

        Returns:
            The negated QUBO, or a copy of it.
        """
        qubo = deepcopy(self) if copy else self

        qubo.is_negated = not qubo.is_negated
        qubo._matrix = -qubo.matrix  # noqa: SLF001
        qubo._offset = -qubo.offset  # noqa: SLF001

        # Clear cached upper and lower bounds
        qubo._upper_bound = None  # noqa: SLF001
        qubo._lower_bound = None  # noqa: SLF001

        return qubo

    def evaluate(self, x: BitVectorLike) -> float:
        """Evaluate the QUBO on a bitvector.

        Args:
            x: Bitvector to evaluate the QUBO on.

        Returns:
            Evaluation of the QUBO on `x`.
        """
        x = BitVector(x)
        return float(x.bits @ self.matrix @ x.bits) + self.offset

    def evaluate_weighted(self, x: Mapping[BitVectorLike, float]) -> float:
        """Evaluate the QUBO on a mapping of bitvectors.

        Args:
            x: Bitvector-weight pairs.

        Returns:
            Weighted sum of evaluations of the QUBO on multiple bitvectors.
        """
        weighted_sum = 0.0
        number_of_shots = sum(x.values())
        for key, frequency in x.items():
            weighted_sum += frequency * self.evaluate(key)
        return weighted_sum / number_of_shots

    def to_string(self, **kwargs: Any) -> str:
        """Format the values in the QUBO matrix as a string.

        Args:
            kwargs: Optional keyword arguments to pass to :py:func:`numpy.array2string`
                formatting.

        Returns:
            Formatted string representation of the QUBO matrix.
        """
        return np.array2string(self.matrix, separator=" ", **kwargs)

    @property
    def size(self) -> int:
        """Size of the QUBO, that is, its number of variables.

        Returns:
            Size of the QUBO.
        """
        return len(self.matrix)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """QUBO matrix."""
        return self._matrix

    @property
    def offset(self) -> float:
        """Offset of the QUBO problem."""
        return self._offset

    @property
    def upper_bound(self) -> float:
        """Upper bound of QUBO."""
        if self._upper_bound is None:
            self.compute_bounds()
            self._upper_bound = cast("float", self._upper_bound)
        return self._upper_bound

    @property
    def lower_bound(self) -> float:
        """Lower bound of QUBO."""
        if self._lower_bound is None:
            self.compute_bounds()
            self._lower_bound = cast("float", self._lower_bound)
        return self._lower_bound

    def eig(self) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Compute the eigenvalues and right eigenvectors of the QUBO matrix.

        This method computes the eigenvalues and right eigenvectors of the QUBO using
        the linalg module of NumPy.

        .. note::
            The eigenvalues and eigenvectors depend on the current form of the QUBO
            matrix. That is, this function may return different results based on whether
            the QUBO matrix is in symmetric form or upper triangular form.

        Returns:
            Eigenvalues and right eigenvectors.

        Raises:
            LinAlgError: If the computation of eigenvalues and eigenvectors does not
                converge.
        """
        return np.linalg.eig(self._matrix)

    def spectral_gap(
        self,
        A: list[float],  # noqa: N803
        B: list[float],  # noqa: N803
        *,
        plot: bool = True,
    ) -> tuple[float, Figure]:
        r"""Create a spectral gap figure and compute the minimum spectral gap.

        Simulates a `quantum annealing process`__ as described by the Hamiltonian

        .. math::

            H_\text{ising} = - \frac{A(s)}{2} \left(\sum_i {\sigma_x^{(i)}}\right)
            + \frac{B(s)}{2} \left(\sum_i h_i {\sigma_z^{(i)}} + \sum_{i > j}
            J_{i, j} {\sigma_z^{(i)}} {\sigma_z^{(j)}}\right)

        where
            - $s \in [0, 1]$ is the time parameter,
            - $A(s)$ and $B(s)$ are the annealing functions,
            - $\sigma_x^{(i)}$ and $\sigma_z^{(i)}$ are Pauli matrices acting on
              qubit $i$,
            - $h_i$ are the qubit biases,
            - $J_{i, j}$ are the coupling strengths.
        The spectral graph figure is a plot of the energy eigenvalues of
        $H_\text{ising}$ over time. The minimum spectral gap is the minimum difference
        between the two lowest energy eigenvalues.

        Not recommended for use when size of QUBO $\ge 10$.

        __  https://docs.dwavesys.com/docs/latest/c_gs_2.html

        Args:
            A: List of values of `$A$` for the annealing process.
            B: List of values of `$B$` for the annealing process.
            plot: Boolean value, if True (default) we plot here. If False we do not plot
                  here.

        Returns:
            The minimum spectral gap and a matplotlib figure containing the spectral gap
            figure.

        Raises:
            ValueError: If A and B do not have equal length, or have length zero.
        """
        if len(A) != len(B):
            msg = f"A and B do not have equal length ({len(A)} and {len(B)})."
            raise ValueError(msg)
        if len(A) == 0:
            msg = "A and B must have positive length."
            raise ValueError(msg)

        external_fields, interactions, _ = self.to_ising()

        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])

        def enclose_in_identity(matrix: NDArray[Any], i: int) -> NDArray[Any]:
            """Encloses the provided matrix in identity matrices.

            Computes the tensor product I_{2^i} ⊗ A ⊗ I_{2^{n - i - 1}},
            corresponding to A acting on qubit i.
            """
            return np.kron(
                np.kron(np.eye(2**i), matrix), np.eye(2 ** (self.size - i - 1))
            )

        # Compute initial and final Lenz-Ising Hamiltonian.
        ham_initial = np.eye(2**self.size) + sum(
            enclose_in_identity(pauli_x, i) for i in range(self.size)
        )
        ham_final = np.eye(2**self.size) + (
            sum(
                external_fields[i] * enclose_in_identity(pauli_z, i)
                for i in range(self.size)
            )
            + sum(
                interactions[i, j]
                * enclose_in_identity(pauli_z, i)
                * enclose_in_identity(pauli_z, j)
                for i in range(self.size)
                for j in range(i + 1, self.size)
            )
        )

        # Compute eigenvalues
        eigs = np.array(
            [
                np.sort(np.linalg.eigvals(-a / 2.0 * ham_initial + b / 2.0 * ham_final))
                for a, b in zip(A, B, strict=False)
            ]
        )

        # Compute minimum gap
        gap = eigs[:, 1] - eigs[:, 0]
        min_gap_idx = gap.argmin()
        min_gap = float(gap[min_gap_idx])

        # Plot spectral graph
        plt.figure()
        plt.title("Spectral graph")
        plt.xlabel("Time")
        plt.ylabel("Energy")

        s = np.linspace(0.0, 1.0, len(A))
        for i in range(2**self.size):
            plt.plot(s, eigs[:, i], color="C0")

        plt.plot([s[min_gap_idx], s[min_gap_idx]], eigs[min_gap_idx, 0:2], color="C1")
        plt.annotate(
            f"minimum gap = {min_gap:.2e}",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(1, -1),
            textcoords="offset fontsize",
            ha="left",
            va="top",
        )

        if plot:
            plt.show()

        return min_gap, plt.gcf()

    def compute_bounds(self, **kwargs: Any) -> NDArray[np.float64]:
        """Compute an upper and lower bound for the QUBO problem.

        First the objective function is convexified and then the integrality constraints
        are dropped (see https://link.springer.com/article/10.1007/s10107-005-0637-9).
        The resulting (convex, constrained) continuous optimization problem is solved
        via TNC algorithm. The solution gives a lower bound. An upper bound is
        obtained by rounding the solution to the relaxation.

        Args:
            kwargs: Optional keyword arguments to pass to TNC algorithm.

        Returns:
            Solution to the convex relaxation of the QUBO.
        """
        # Issue warning if function has been called before and
        # no parameters are given
        if (
            (self._upper_bound is not None)
            and (self._lower_bound is not None)
            and (not kwargs)
        ):
            warnings.warn(
                "QUBO bounds are being recomputed "
                "but no optimization arguments were passed",
                stacklevel=1,
            )

        # Special case: QUBO size zero
        if self.size == 0:
            self._lower_bound = self.offset
            self._upper_bound = self.offset
            return np.array([], dtype=np.float64)

        # Make sure matrix is symmetric
        self.to_symmetric_form()

        # Convexification step
        eigenvalues, _ = self.eig()  # Note: Q is symmetric so eigenvalues are real
        lambda_min = np.amin(eigenvalues)
        shift = min(np.floor(lambda_min), 0)
        qmat_conv = self.matrix - shift * np.identity(self.size)
        q_conv = shift * np.ones(self.size)

        # Variable bounds 0 <= x <= 1
        bounds = Bounds([0.0] * self.size, [1.0] * self.size)

        # Define function to optimize
        def qubo_fun(
            x: NDArray[np.float64], qmat: NDArray[np.float64], q: NDArray[np.float64]
        ) -> float:
            return float(x @ qmat @ x + q @ x)

        def grad_qubo_fun(
            x: NDArray[np.float64], qmat: NDArray[np.float64], q: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return 2.0 * qmat @ x + q

        # Optimize
        rng = np.random.RandomState(seed=0)
        x0 = rng.randint(low=0, high=2, size=self.size).astype(np.float64)
        options = {"disp": False}
        options.update(kwargs)
        res = minimize(
            qubo_fun,
            x0,
            method="TNC",
            jac=grad_qubo_fun,
            args=(qmat_conv, q_conv),
            options=options,
            bounds=bounds,
        )
        x_relax = np.array(res.x)

        # Get bounds
        self._lower_bound = qubo_fun(x_relax, qmat_conv, q_conv) + self.offset
        x_feas = np.round(x_relax)
        self._upper_bound = self.evaluate(x_feas)

        return x_relax

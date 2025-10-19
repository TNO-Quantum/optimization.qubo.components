"""Module that contains the `PartialSolution` class."""

from __future__ import annotations

import re
from typing import Any, SupportsIndex, SupportsInt

import numpy as np
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import Serializable
from tno.quantum.utils.validation import check_binary, check_bool, check_int

from tno.quantum.optimization.qubo.components._qubo import QUBO


class PartialSolution(Serializable):  # noqa: PLW1641
    """Class representing a partial solution to a QUBO.

    A partial solution is a collection of assignments of variables of the QUBO of one of
    the following forms:

    - $x_i := x_j$ `(variable-to-variable)`,
    - $x_i := 1 - x_j$ `(variable-to-conjugate-variable)`,
    - $x_i := 0$ or $x_i := 1$ `(variable-to-value)`.

    Example:
        >>> from tno.quantum.optimization.qubo.components import PartialSolution
        >>>
        >>> # Create a PartialSolution in `n = 5` variables
        >>> n = 5
        >>> p = PartialSolution(n)
        >>>
        >>> # Make some assignments
        >>> p.assign_variable(0, 1)             # x_0 := x_1
        >>> p.assign_variable(2, 3, conj=True)  # x_2 := 1 - x_3
        >>> p.assign_value(3, 1)                # x_3 := 1
        >>>
        >>> # Show which of the `n = 5` variables are unassigned
        >>> p.free_variables()
        [1, 4]
        >>> # Suppose we find the free variables x_1 and x_4 should have values 0 and 1.
        >>> # The full solution (x_0, ... , x_4) is obtained via p.expand
        >>> p.expand([1, 0]) # (x_1, x_4) = (1, 0)
        BitVector(11010)
    """

    def __init__(self, n: SupportsInt) -> None:
        """Init :py:class:`PartialSolution`.

        Args:
            n: Number of variables of the QUBO.
        """
        self._n = int(n)
        # `self.assignments[i]` is a pair `(j, conj_j)` indicating that variable
        # x_i is assigned the value x_j (if !conj_j) or 1 - x_j (if conj_j)
        self._assignments: dict[int, tuple[int, bool]] = {}

    @property
    def n(self) -> int:
        """Number of variables of the QUBO."""
        return self._n

    def assign_variable(
        self, i: SupportsIndex, j: SupportsIndex, *, conj: bool = False
    ) -> None:
        """Assigns variable $x_i$ to variable $x_j$ or its conjugate.

        Args:
            i: Index of the variable to be assigned.
            j: Index of the variable to which it is assigned.
            conj: `True` if $x_i$ is set to $1 - x_j$, `False` if it is set to $x_j$.

        Raises:
            ValueError: If `i` or `j` does not lie in the range $[0, .. , n - 1]$, or
                if $x_i$ is already assigned a value, or if the assignment creates an
                assignment cycle.
        """
        i = check_int(i, "i", l_bound=0, u_bound=self._n - 1)
        j = check_int(j, "j", l_bound=0, u_bound=self._n - 1)
        conj = check_bool(conj, "conj", safe=True)

        self._check_already_assigned(i)
        self._check_assignment_cycle(i, j)

        self._assignments[i] = (j, conj)

    def _check_already_assigned(self, i: int) -> None:
        if i in self._assignments:
            msg = f"Variable x_{i} is already assigned"
            raise ValueError(msg)

    def _check_assignment_cycle(self, i: int, j: int) -> None:
        """Check for assignment cycles.

        Checks if if assigning x_i to x_j (or its conjugate) will result in an
        assignment cycle.
        """
        # If a cycle will be created due to assigning i to j, then j must currently
        # already resolve to i. If this is the case we raise an error. Also if j == i,
        # a cycle will be created.
        if j == i or self._resolve(j)[0] == i:
            msg = "Assignment creates a cycle"
            raise ValueError(msg)

    def assign_value(self, i: SupportsIndex, value: SupportsInt) -> None:
        """Assigns variable $x_i$ the value `value`.

        Args:
            i: Index of the variable to be assigned.
            value: Value to which it is assigned. Must be 0 or 1.

        Raises:
            ValueError: If `i` or `j` does not lie in the range $[0, .. , n - 1]$, or
                if $x_i$ was already assigned a value, or if `value` does not equal
                0 or 1.
        """
        i = check_int(i, "i", l_bound=0, u_bound=self._n - 1)
        value = check_binary(value, "value")

        self._check_already_assigned(i)

        # Trick: to indicate that x_i is assigned a constant value, we introduce a
        # 'virtual variable' x_{-1} = 0, with conjugate 1 - x_{-1} = 1.
        self._assignments[i] = (-1, value > 0)

    def expand(self, x: BitVectorLike) -> BitVector:
        """Expands a bitvector assuming the partial solution.

        Given a bitvector in terms of the free variables, convert it to a bitvector in
        terms of the initial `n` variables, assuming the partial solution.

        Args:
            x: Bitvector in terms of the free variables.

        Returns:
            Bitvector in terms of the initial `n` variables.

        Raises:
            ValueError: If the length of `x` does not match the number of free
                variables.
        """
        x = BitVector(x)
        free_variables = self.free_variables()

        if len(x) != len(free_variables):
            msg = f"Bitvector has length {len(x)}, expected {len(free_variables)}"
            raise ValueError(msg)

        # Compute output bit vector of length n
        y = np.zeros(self._n, dtype=np.uint8)
        for j in range(self._n):
            (k, conj) = self._resolve(j)
            if k == -1:  # Index -1 indicates a 'virtual variable' x_{-1} = 0
                y[j] = 1 if conj else 0
            else:
                i = free_variables.index(k)
                y[j] = 1 - x[i] if conj else x[i]

        return BitVector(y)

    def free_variables(self) -> list[int]:
        """List of indices of unassigned variables.

        Returns:
            List of indices of the variables which are not assigned.
        """
        return [i for i in range(self._n) if i not in self._assignments]

    def apply(self, qubo: QUBO) -> QUBO:
        """Reduces QUBO assuming the partial solution.

        Assuming the assignments of this partial solution, create a lower-dimensional
        QUBO representing an equivalent problem (in terms of the free variables).

        Args:
            qubo: QUBO to be reduced.

        Returns:
            Reduced QUBO.

        Raises:
            ValueError: If the size of the QUBO does not equal `n`.
        """
        if qubo.size != self._n:
            msg = f"QUBO has size {qubo.size}, expected {self._n}"
            raise ValueError(msg)

        free_variables = self.free_variables()
        m = len(free_variables)

        # Initialize matrix and offset for reduced QUBO
        Q = np.zeros((m, m))  # noqa: N806
        offset = qubo.offset

        # For every term Q_ij x_i x_j of the original QUBO, compute the contribution to
        # the reduced QUBO. This contribution can be written as:
        #
        #   c_uv x_u x_v + c_u x_u + c_v x_v + c
        #
        # where x_i resolves to x_u (or 1 - x_u) and x_j resolves to x_v (or 1 - x_v).
        # The values of the coefficients c_uv, c_u, c_v and c thus depend on Q_ij
        # depending on `conj_u` and `conj_v`. Example: if `conj_u = conj_v = True`, then
        #
        #   Q_ij x_i x_j = Q_ij (1 - x_u) (1 - x_v)
        #                = Q_ij x_u x_v - Q_ij x_u - Q_ij x_v + Q_ij
        for i in range(self._n):
            (u, conj_u) = self._resolve(i)
            if u != -1:
                u = free_variables.index(u)
            for j in range(self._n):
                (v, conj_v) = self._resolve(j)
                if v != -1:
                    v = free_variables.index(v)

                q_ij = qubo.matrix[i, j]

                # The index `-1` indicates the the 'virtual variable' x_{-1} = 0.
                # Hence, if u = -1 (resp v = -1), we immediately evaluate x_u (resp x_v)
                # to zero, dropping the corresponding terms.
                if u != -1 and v != -1:
                    c_uv = -q_ij if conj_u ^ conj_v else q_ij
                    Q[u, v] += c_uv
                if u != -1:
                    c_u = 0 if not conj_v else -q_ij if conj_u else q_ij
                    Q[u, u] += c_u
                if v != -1:
                    c_v = 0 if not conj_u else -q_ij if conj_v else q_ij
                    Q[v, v] += c_v
                c = q_ij if conj_u and conj_v else 0
                offset += c

        return QUBO(Q, offset)

    def _resolve(self, i: int) -> tuple[int, bool]:
        """Resolves $x_i$ to some $x_j$ (or its conjugate), returning ``(j, conj)``."""
        # Base case: if x_i is not assigned, it resolves to itself
        if i not in self._assignments:
            return (i, False)
        # If x_i is assigned to x_j (or its conjugate), resolve x_j
        (j, conj_j) = self._assignments[i]
        (k, conj_k) = self._resolve(j)
        # Overall conjugate factor is XOR of the two conjugate factors
        return (k, conj_j ^ conj_k)

    def __repr__(self) -> str:
        assigns = []
        for i in range(self._n):
            if i not in self._assignments:
                continue
            (u, conj_u) = self._resolve(i)
            if u == -1:
                assigns.append(f"x_{i} = {'1' if conj_u else '0'}")
            else:
                assigns.append(f"x_{i} = {'1 - ' if conj_u else ''}x_{u}")
        return self.__class__.__name__ + " {" + ", ".join(assigns) + "}"

    def _serialize(self) -> dict[str, str | int]:
        """Serialize to dict."""
        data: dict[str, str | int] = {}
        data["n"] = self._n
        for i, (j, conj_j) in self._assignments.items():
            if j == -1:
                data[f"x{i}"] = 1 if conj_j else 0
            else:
                data[f"x{i}"] = f"1 - x{j}" if conj_j else f"x{j}"
        return data

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> PartialSolution:
        """Deserialize from dict."""
        n = data.pop("n", None)

        if n is None:
            msg = "Missing key 'n'"
            raise KeyError(msg)

        if not isinstance(n, int):
            msg = f"Invalid value '{n}' for 'n'"
            raise TypeError(msg)

        partial_solution = PartialSolution(n)

        for key, value in data.items():
            m = re.search(r"^x(\d+)$", key)
            if m is None:
                msg = f"Invalid key '{key}'"
                raise KeyError(msg)

            i = int(m.group(1))

            if isinstance(value, int):
                partial_solution.assign_value(i, value)
            elif m := re.search(r"^x(\d+)$", value):
                partial_solution.assign_variable(i, int(m.group(1)))
            elif m := re.search(r"^1 - x(\d+)$", value):
                partial_solution.assign_variable(i, int(m.group(1)), conj=True)
            else:
                msg = f"Invalid value '{value}'"
                raise ValueError(msg)

        return partial_solution

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PartialSolution):
            return False

        return self._n == other._n and all(
            self._resolve(i) == other._resolve(i)  # noqa: SLF001
            for i in range(self._n)
        )

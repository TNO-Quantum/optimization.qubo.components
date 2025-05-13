"""Utility function to create QUBO Solver instances."""

from __future__ import annotations

import inspect
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tno.quantum.optimization.qubo.components._solvers._solver import Solver
from tno.quantum.utils import BaseConfig, get_installed_subclasses


@dataclass(init=False)
class SolverConfig(BaseConfig[Solver[Any]]):
    """Configuration class for creating an instance of solver.

    Example:
        (Requires :py:mod:`tno.quantum.optimization.qubo.solvers` to be installed.)

        >>> from tno.quantum.optimization.qubo.components import SolverConfig
        >>> list(SolverConfig.supported_items())  # doctest: +SKIP
        ['bf_solver',
        'custom_solver',
        'd_wave_clique_sampler_solver',
        'd_wave_sampler_solver',
        'exact_sampler_solver',
        'kerberos_sampler_solver',
        'leap_hybrid_solver',
        'neighborhood_solver',
        'pipeline_solver',
        'qaoa_solver',
        'rs_solver',
        'random_sampler_solver',
        'sa2_solver',
        'simulated_annealing_solver',
        'steepest_descent_solver',
        'tabu_solver',
        'tree_decomposition_solver']
        >>> config = SolverConfig(name="bf_solver")  # doctest: +SKIP
        >>> config.get_instance()  # doctest: +SKIP
        <...BFSolver...>
    """

    def __init__(self, name: str, options: Mapping[str, Any] | None = None) -> None:
        """Init :py:class:`SolverConfig`.

        Args:
            name: Name of the solver class.
            options: Keyword arguments to be passed to the solver. Must be a
                mapping-like object whose keys are strings, and whose values can be
                anything depending on specific solver.

        Raises:
            TypeError: If `name` is not an string, or if `options` is not a mapping.
            KeyError: If `options` has a key that is not a string.
            ValueError: If the `supported_items` method returns a dict with keys that
                do not adhere to the snake_case convention.
        """
        super().__init__(name=name, options=options)

    @staticmethod
    def supported_items() -> dict[str, type[Solver[Any]]]:
        """Returns dictionary of supported solvers.

        Finds all implementations of :py:class:`Solver` in the installed
        submodules of :py:mod:`tno.quantum.optimization.qubo`.

        Returns:
            Dictionary with solvers by their name in snake-case .
        """
        supported_solvers: dict[str, type[Solver[Any]]] = {}

        # Discover all submodules in the qubo package
        from tno.quantum.optimization import qubo

        qubo_path = [str(path) for path in qubo.__path__]
        submodules = [name for _, name, _ in pkgutil.iter_modules(qubo_path)]
        base_path = "tno.quantum.optimization.qubo"
        for submodule in submodules:
            # Get all installed solvers
            installed_solvers = get_installed_subclasses(
                f"{base_path}.{submodule}", subclass=Solver
            )

            # Remove all abstract classes
            installed_solvers = {
                name: class_obj
                for name, class_obj in installed_solvers.items()
                if not inspect.isabstract(class_obj)
            }

            supported_solvers.update(installed_solvers)

        return supported_solvers

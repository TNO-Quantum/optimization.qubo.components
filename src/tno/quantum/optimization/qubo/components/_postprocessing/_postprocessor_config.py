"""This module contains the ``PostprocessorConfig`` configuration class."""

from __future__ import annotations

import inspect
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from tno.quantum.utils import BaseConfig, get_installed_subclasses

from tno.quantum.optimization.qubo.components._postprocessing._postprocessor import (
    Postprocessor,
)
from tno.quantum.optimization.qubo.components._results._result_interface import (
    ResultInterface,
)

RESULT_TYPE = TypeVar("RESULT_TYPE", bound=ResultInterface)


@dataclass(init=False)
class PostprocessorConfig(BaseConfig[Postprocessor[Any]]):
    """Configuration class for creating an instance of a postprocessor.

    Example:
        (Requires :py:mod:`tno.quantum.optimization.qubo.postprocessors` to be installed.)

        >>> from tno.quantum.optimization.qubo.components import PostprocessorConfig
        >>> list(PostprocessorConfig.supported_items())  # doctest: +SKIP
        ['steepest_descent_postprocessor']
        >>> config = PostprocessorConfig(name="steepest_descent_postprocessor", options={})   # doctest: +SKIP
        >>> config.get_instance()  # doctest: +SKIP
        <...SteepestDescentPostprocessor...>
    """  # noqa: E501

    def __init__(self, name: str, options: Mapping[str, Any] | None = None) -> None:
        """Init :py:class:`PostprocessorConfig`.

        Args:
            name: Name of the postprocessor class.
            options: Keyword arguments to be passed to the postprocessor. Must be a
                mapping-like object whose keys are strings, and whose values can be
                anything depending on specific postprocessor.

        Raises:
            TypeError: If `name` is not an string, or if `options` is not a mapping.
            KeyError: If `options` has a key that is not a string.
            ValueError: If the `supported_items` method returns a dict with keys that
                do not adhere to the snake_case convention.
        """
        super().__init__(name=name, options=options)

    @staticmethod
    def supported_items() -> dict[str, type[Postprocessor[Any]]]:
        """Returns dictionary of supported postprocessors.

        Finds all implementations of :py:class:`Postprocessor` in the installed
        submodules of :py:mod:`tno.quantum.optimization.qubo`.

        Returns:
            Dictionary with postprocessors by their name in snake-case.
        """
        supported_postprocessors: dict[str, type[Postprocessor[Any]]] = {}

        # Discover all submodules in the qubo package
        from tno.quantum.optimization import qubo  # noqa: PLC0415

        qubo_path = [str(path) for path in qubo.__path__]
        submodules = [name for _, name, _ in pkgutil.iter_modules(qubo_path)]
        base_path = "tno.quantum.optimization.qubo"
        for submodule in submodules:
            # Get all installed postprocessors
            installed_postprocessors = get_installed_subclasses(
                f"{base_path}.{submodule}", subclass=Postprocessor
            )

            # Remove all abstract classes
            installed_postprocessors = {
                name: class_obj
                for name, class_obj in installed_postprocessors.items()
                if not inspect.isabstract(class_obj)
            }

            supported_postprocessors.update(installed_postprocessors)

        return supported_postprocessors

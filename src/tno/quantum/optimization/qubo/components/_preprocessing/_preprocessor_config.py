"""Module that contains the :class:`PreprocessorConfig` configuration class."""

from __future__ import annotations

import inspect
import pkgutil
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tno.quantum.optimization.qubo.components._preprocessing._preprocessor import (
    Preprocessor,
)
from tno.quantum.utils import BaseConfig, get_installed_subclasses


@dataclass(init=False)
class PreprocessorConfig(BaseConfig[Preprocessor]):
    """Configuration class for creating an instance of a preprocessor.

    Example:
        (Requires :py:mod:`tno.quantum.optimization.qubo.preprocessors` to be installed.)

        >>> from tno.quantum.optimization.qubo.components import PreprocessorConfig
        >>> list(PreprocessorConfig.supported_items())  # doctest: +SKIP
        ['q_pro_plus_preprocessor']
        >>> config = PreprocessorConfig(name="q_pro_plus_preprocessor", options={"max_iterations": 10})  # doctest: +SKIP
        >>> config.get_instance()  # doctest: +SKIP
        <...QProPlusPreprocessor...>
    """  # noqa: E501

    def __init__(self, name: str, options: Mapping[str, Any] | None = None) -> None:
        """Init :py:class:`PreprocessorConfig`.

        Args:
            name: Name of the preprocessor class.
            options: Keyword arguments to be passed to the preprocessor. Must be a
                mapping-like object whose keys are strings, and whose values can be
                anything depending on specific preprocessor.

        Raises:
            TypeError: If `name` is not an string, or if `options` is not a mapping.
            KeyError: If `options` has a key that is not a string.
            ValueError: If the `supported_items` method returns a dict with keys that
                do not adhere to the snake_case convention.
        """
        super().__init__(name=name, options=options)

    @staticmethod
    def supported_items() -> dict[str, type[Preprocessor]]:
        """Returns dictionary of supported preprocessors.

        Finds all implementations of :py:class:`Preprocessor` in the installed
        submodules of :py:mod:`tno.quantum.optimization.qubo`.

        Returns:
            Dictionary with preprocessors by their name in snake-case .
        """
        supported_preprocessors: dict[str, type[Preprocessor]] = {}

        # Discover all submodules in the qubo package
        from tno.quantum.optimization import qubo

        qubo_path = [str(path) for path in qubo.__path__]
        submodules = [name for _, name, _ in pkgutil.iter_modules(qubo_path)]
        base_path = "tno.quantum.optimization.qubo"
        for submodule in submodules:
            # Get all installed preprocessors
            installed_preprocessors = get_installed_subclasses(
                f"{base_path}.{submodule}", subclass=Preprocessor
            )

            # Remove all abstract classes
            installed_preprocessors = {
                name: class_obj
                for name, class_obj in installed_preprocessors.items()
                if not inspect.isabstract(class_obj)
            }

            supported_preprocessors.update(installed_preprocessors)

        return supported_preprocessors

"""This package contains the components to define QUBOs and solvers.

**Example**

The following example shows how to construct a
:py:class:`~tno.quantum.optimization.qubo.components.QUBO` object.

>>> from tno.quantum.optimization.qubo.components import QUBO
>>> qubo = QUBO([
...     [ 1, -2,  3],
...     [-4,  5, -6],
...     [ 7, -8,  9]
... ])

One can use the :py:class:`~tno.quantum.optimization.qubo.components.SolverConfig` class
to find and instantiate available solvers to solve this QUBO, as shown in the following
example. (Note that this requires the :py:mod:`tno.quantum.optimization.qubo.solvers`
package to be installed.)

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
>>> solver = SolverConfig(name='bf_solver').get_instance()  # doctest: +SKIP

>>> # Solve the QUBO
>>> result = solver.solve(qubo)  # doctest: +SKIP
>>> result.best_bitvector  # doctest: +SKIP
BitVector(000)

Similarly, one can use the
:py:class:`~tno.quantum.optimization.qubo.components.PreprocessorConfig` and
:py:class:`~tno.quantum.optimization.qubo.components.PostprocessorConfig` classes to
find and instantiate available preprocessors
(:py:class:`~tno.quantum.optimization.qubo.components.Preprocessor`) and postprocessors
(:py:class:`~tno.quantum.optimization.qubo.components.Postprocessor`).
"""

from tno.quantum.optimization.qubo.components._freq import Freq
from tno.quantum.optimization.qubo.components._postprocessing._postprocessor import (
    Postprocessor,
)
from tno.quantum.optimization.qubo.components._postprocessing._postprocessor_config import (  # noqa: E501
    PostprocessorConfig,
)
from tno.quantum.optimization.qubo.components._preprocessing._partial_solution import (
    PartialSolution,
)
from tno.quantum.optimization.qubo.components._preprocessing._preprocessor import (
    Preprocessor,
)
from tno.quantum.optimization.qubo.components._preprocessing._preprocessor_config import (  # noqa: E501
    PreprocessorConfig,
)
from tno.quantum.optimization.qubo.components._qubo import QUBO
from tno.quantum.optimization.qubo.components._results._basic_result import (
    BasicResult,
)
from tno.quantum.optimization.qubo.components._results._result_interface import (
    ResultInterface,
)
from tno.quantum.optimization.qubo.components._solvers._solver import Solver
from tno.quantum.optimization.qubo.components._solvers._solver_config import (
    SolverConfig,
)

__all__ = [
    "QUBO",
    "BasicResult",
    "Freq",
    "PartialSolution",
    "Postprocessor",
    "PostprocessorConfig",
    "Preprocessor",
    "PreprocessorConfig",
    "ResultInterface",
    "Solver",
    "SolverConfig",
]

__version__ = "1.0.0"

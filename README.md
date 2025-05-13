# TNO Quantum: Optimization - QUBO - Components

TNO Quantum provides generic software components aimed at facilitating the development
of quantum applications.

This package contains the components to define QUBOs and solvers.

## Documentation

Documentation of the `tno.quantum.optimization.qubo.components` package can be found [here](https://tno-quantum.github.io/documentation/).


## Install

Easily install the `tno.quantum.optimization.qubo.components` package using pip:

```console
$ python -m pip install tno.quantum.optimization.qubo.components
```


## Usage

The QUBO Components package can be used to define custom solver classes as shown in the following example.

```python
from tno.quantum.optimization.qubo.components import QUBO, Solver, BasicResult

class CustomSolver(Solver[BasicResult]):
   def _solve(self, qubo: QUBO) -> BasicResult:
      result = ... # solve QUBO and construct result
      return result
```

The example below shows how to obtain all installed solvers.

An instance of a solver can be obtained via the `get_instance()` function on an `SolverConfig` instance.

*Note:* the `"simulated_annealing_solver"` solver shown in the example requires `tno.quantum.optimization.qubo.solvers` to be installed.  

```python
from tno.quantum.optimization.qubo.components import SolverConfig

supported_solvers = SolverConfig.supported_items()

solver_config = SolverConfig(name="simulated_annealing_solver", options={})
solver = solver_config.get_instance()
```

## (End)use limitations
The content of this software may solely be used for applications that comply with international export control laws.

# 1.2.0 (2025 - 10 - 19)

Support for `python3.9` is dropped.

### Features

* **Repr:** default string and HTML representations of `ResultInterface`
* **QUBO:** optionally allow for sorting of labels when calling `QUBO.from_bqm`

### Fixes

* Fixed `compute_bounds` for empty QUBO and speed improvements.
* Bugfix QUBO lower bound now takes offset into account.

# 1.0.0 (2025 - 05 - 13)

Initial public release.

### Features

* **QUBO:** Class representing a Quadratic Unconstrained Binary Optimization  problem.
* **Solver output classes:** Basic result objects for solvers such as `Freq`, `BasicResult`, `ResultInterface`, `PartialSolution`
* **Base classes:** Base classes for `Solver`, `Postprocessor` and `Preprocessor`.
* **Configs:** Introduce `SolverConfig`, `PreprocessorConfig`, `PostprocessorConfig`


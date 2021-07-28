Source code for a study of dynamic parameter learning from partial observations of the Lorenz eqautions.

## Prerequisites
The packages [`alphashape`](https://pypi.org/project/alphashape/) and [`descartes`](https://pypi.org/project/descartes/) are used to produce bounding polygons in the figures.

## Contents
### Code
* [`estimate.py`](https://github.com/unis-ing/lorenz_parameter_learning/blob/main/estimate.py): Defines a function `estimate()` for estimating the Lorenz parameters from continuous observations of the solution.
* [`save.py`](https://github.com/unis-ing/lorenz_parameter_learning/blob/main/save.py): Helper functions for managing simulation data.
* [`plot.py`](https://github.com/unis-ing/lorenz_parameter_learning/blob/main/plot.py): Helper functions for producing the parameter sweep figures.
* [`lorenz_AOT_simple.m`](https://github.com/unis-ing/lorenz-parameter-learning/blob/main/lorenz_AOT_simple.m): A MATLAB script for estimating the Lorenz parameters from discrete-in-time observations of the solution. The amplitude of measurement noise and stochastic forcing can be controlled through parameters `eta` and `epsilon`.

### Notebooks
* [`demo.ipynb`](https://github.com/unis-ing/lorenz_parameter_learning/blob/main/demo.ipynb): Example usage of `estimate()`.
* [`figures.ipynb`](https://github.com/unis-ing/lorenz_parameter_learning/blob/main/figures.ipynb): Code for producing the parameter sweep figures found in the paper.

### Data
* [`data/sigma/`](https://github.com/unis-ing/lorenz-parameter-learning/tree/main/data/sigma): Continuously estimating σ.
* [`data/sigma_translated/`](https://github.com/unis-ing/lorenz-parameter-learning/tree/main/data/sigma_translated): Continuously estimating σ using an alternate, "translated" recovery formula.
* [`data/rho_beta/`](https://github.com/unis-ing/lorenz-parameter-learning/tree/main/data/rho_beta): Continuously estimating ρ and β.
* [`data/rho_sigma/`](https://github.com/unis-ing/lorenz-parameter-learning/tree/main/data/rho_sigma): Continuously estimating σ and ρ.
* [`data/sigma_beta/`](https://github.com/unis-ing/lorenz-parameter-learning/tree/main/data/sigma_beta): Continuously estimating σ and β.

## Authors
* Adam Larios (University of Nebraska-Lincoln)
* Eunice Ng (Hunter College)

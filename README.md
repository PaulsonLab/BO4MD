# BO4MD

## Overview
Bayesian Optimization for Automated Molecular Dynamics Simulations

## Installation
1. **Clone the repository:**
```sh
git clone https://github.com/PaulsonLab/BO4MD.git
```
2. **Navigate to `~/bo4md`:**
```sh
cd ./bo4md
```
3. **Install via [poetry](https://python-poetry.org/):**
```sh
poetry install
```

## Command-Line Usage

You can run Bayesian Optimization directly from the command line:
```sh
python -m bo4md --smoke_test true --acq logei --d 3 --n-init 10 --n-iter 20 --patience 5 --seed 42 --plot true --report true --outfolder /out
```
### Input Arguments

- `--smoke_test` (bool, default: true)  
  If true, runs the built-in synthetic smoke test function.  
  If false, runs the MD simulator function (must be implemented by the user).

- `--acq` (str, default: "logei")  
  Acquisition function. Choices: ucb, ei, logei, random.

- `--d` (int, default: 3)  
  Input dimensionality (number of simplex components).

- `--n-init` (int, default: 10)  
  Number of initial random samples drawn from the simplex.

- `--n-iter` (int, default: 20)  
  Maximum number of Bayesian Optimization iterations.

- `--patience` (int, default: 5)  
  Early stopping patience (stop if no improvement for this many iterations).

- `--seed` (int or None, default: None)  
  Random seed for reproducibility.

- `--plot` (bool, default: true)  
  If true, generates a plot of best-so-far objective value vs. iteration (bo_traj.png).

- `--report` (bool, default: true)  
  If true, writes a text report of the optimization trajectory (out.txt).

- `outfolder` (str or None, default: ./out)
  Output folder for initial data file, trajectory plot, and report.

### Examples

Run a quick smoke test with default settings:
```sh
python -m bo4md
```

Run a smoke test with 30 iterations using UCB acquisition, 15 initial samples, and a fixed seed:
```sh
python -m bo4md --smoke_test true --acq ucb --n-init 15 --n-iter 30 --seed 123
```

Run with the MD simulator instead of the smoke test:
```sh
python -m bo4md --smoke_test false 
```

## Note
 The *md(.)* function in [`simulator.py`](./src/bo4md/simulator.py) serves as a template which needs to be adapted to call real MD simulations. It expects an [n, d] torch tensor as input and returns an [n] torch tensor (simulation output), where n is number of data points and d is input dimension.

## Citation
```
TBD
```

# BO4MD

## Overview
Bayesian Optimization for Molecular Dynamics Simulation Automation

## Installation
1. **Clone the repository:**
```sh
git clone https://github.com/PaulsonLab/BO4MD.git
```
2. **Navigate to `./bo4md`:**
```sh
cd ./bo4md
```
3. **Install via [poetry](https://python-poetry.org/):**
```sh
poetry install
```


## Usage
Run the following in the command line:
```sh
poetry run python bayes_opt.py --smoke_test True --acq logei --n-init 15 --n-iter 30 --patience 10 --seed 42
```
Or
```sh
python bayes_opt.py --smoke_test True --acq logei --n-init 15 --n-iter 30 --patience 10 --seed 42
```

The inputs are: 
- `file` - filename of the SEM image (.csv, .png, .tif, and .tiff are supported)
- `tol` (default=0.20) - threshold to compare against the relative difference between the feature sizes computed based on an average of all directions and the dominant direction, which is used to determine whether the features are isotropic or not
- `plot` (True or False, default=True) - whether to plot out the radial profile of the autocorrelation map
- `xlim` (default=None) - range of radius for the radial profile plot for the purpose of improving visualization of the tiny peak
- `ylim` (default=None) - range of autocorrelation for the radial profile plot for the purpose of improving visualization of the tiny peak

## Citation
```
TBD
```

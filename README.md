# Partial Generalized Coherence (PGC)
This repository contains code underlying our paper on partial generalized coherence (PGC), which quantifies frequency coupling between two time series conditioned on the activity of other time series without making any model assumptions. Furthermore, PGC is not limited to same-frequency coupling resulting from linear interactions but also can address cross-frequency coupling resulting from nonlinear interactions. Please see the paper for relevant references.

## Installation
PGC equires the following packages:
- NumPy
- SciPy
- seaborn
- TensorFlow (<v2.0)

PGC also requires [CCMI](https://github.com/sudiptodip15/CCMI). Please place the `CIT` folder of CCMI into the root directory of PGC, i.e. at the same directory level as this README.

## Usage
The `PGC.py` file contains all functions used for PGC. The core functions are:
- `pcoh`: Computes the partial coherence between time series.
- `pgc`: Can compute MIF and PGC between time series.

For more detail, view the header for each function in the `PGC.py` file. Also view the following examples to see how to use the functions in practice.

## Examples Used in Paper
### 1. Linear 3-node example (figure 3) ([Jupyter Notebook](figure3.ipynb))
A simple linear gaussian network is analyzed using partial coherence and PGC.

### 2. Nonlinear 3-node example (figure 4) ([Jupyter Notebook](figure4.ipynb))
A nonlinear network is analyzed using PGC.

### 3. Scaling (figure 5) ([Jupyter Notebook](figure5.ipynb))
A demonstration of how the scaling performance of PGC can be evaluated using Gaussian distributions, which underlies the production of figure 5.

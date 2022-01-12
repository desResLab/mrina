[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![example workflow](https://github.com/desResLab/mrina/actions/workflows/test_publish_pypi.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/mrina/badge/?version=latest)](https://mrina.readthedocs.io/en/latest/?badge=latest)

# MRIna: A library for MRI Noise Analysis

MRIna is a library for the analysis of reconstruction noise from undersampled 4D flow MRI. For additional details, please refer to the publication below:

Lauren Partin, [Daniele E. Schiavazzi](https://www3.nd.edu/~dschiava/) and [Carlos A. Sing-Long Collao](https://www.ing.uc.cl/academicos-e-investigadores/carlos-alberto-sing-long-collao/), *An analysis of reconstruction noise from undersampled 4D flow MRI* [arXiv](http://arxiv.org/abs/2201.03715)

The complete set of results from the above paper can be found [at this link](https://notredame.box.com/s/fdrd3e3du555u1ikarrfkvt3jsxddwe9)

---

## Installation and documentation

You can install MRIna with pip ([link to PyPI](https://pypi.org/project/mrina/))
```
pip install PyWavelets mrina
```

For the documentation follow this [link](https://mrina.readthedocs.io/en/latest/).

---

## What you can do with MRIna.

The MRIna library provides the following functionalities.

- It generates k-space undersampling masks of various types including **Bernoulli**, **variable density triangular**, **variable density Gaussian**, **variable density exponential** and **Halton** quasi-random sequences. 
- It supports arbitrary operators that implement a forward call (**eval**), and inverse call (**adjoint**), column restriction (**colRestrict**), **shape** and **norm**.
- It supports various **non-linear reconstruction methods** including l1-norm minimization with iterative thresholding and orthogonal matching pursuit based greedy heuristics.
- It provides a number of scripts to 
 
  + generate ensembles of synthetic, subsampled and noisy k-space images (4 complex images);
  + reconstruct image density and velocities;
  + post-process to compute correlations, MSE, error patterns and relative errors.

---

## Core Dependencies
* Python 3.6.5
* [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) 1.1.1
* [Numpy](https://numpy.org/) 1.18.1
* [Scipy](https://www.scipy.org/) 1.1.0
* [Matplotlib](https://matplotlib.org/) 3.1.0
* [Cython](https://cython.org/)
* [opencv](https://opencv.org/)

## Citation
Did you find this useful? Cite us using:
```
@misc{partin2022analysis,
      title={An analysis of reconstruction noise from undersampled 4D flow MRI}, 
      author={Lauren Partin and Daniele E. Schiavazzi and Carlos A. Sing Long},
      year={2022},
      eprint={2201.03715},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```


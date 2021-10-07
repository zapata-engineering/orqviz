# orqviz
A Python package for easily **visualizing the loss landscape** of Variational Quantum Algorithms by Zapata Computing, Inc. 

`orqviz` provides a collection of tools which quantum researchs and enthusiasts alike can use for their simulations. It works with any framework for running quantum circuits, for example `qiskit`, `cirq`, and `pennylane`. The package contains functions to generate data, as well as plotting functions and helpers to plot them flexibly. The package is light-weight and has very few dependencies. 

## Getting started

In our repository you will find an [example Jupter notebook](https://github.com/zapatacomputing/orqviz/blob/main/examples/example_notebook.ipynb) providing a thorough demonstration of the tools available with the package. 
In [this Github](https://github.com/zapatacomputing/visualization-tutorials) repository we have four Jupyter notebooks with tutorials for how to get started with any quantum circuit simulation framework you might use. You will find examples with `qiskit`, `cirq`, `pennylane` and Zapata's `Orquestra` library. The tutorials are not exhaustive, but they do provide a full story that you can follow along.

We have recently published a paper on arXiv where we review the tools available with `orqviz`. TODO: Link to arXiv.

## Installation
You can install our package like
```bash
pip install orqviz
```

Alternatively you can build the package from source. This is especially helpful if you would like to contribute to `orqviz`
```bash
git clone https://github.com/zapatacomputing/orqviz.git
cd orqviz
pip install -e ./
```

## Examples
```python
import orqviz
import numpy as np

np.random.seed(42)

def loss_function(pars):
    return np.sum(np.cos(pars))**2

n_params = 42
params = np.random.uniform(-np.pi, np.pi, size=n_params)
dir1 = orqviz.geometric.get_random_normal_vector(n_params)
dir2 = orqviz.geometric.get_random_orthonormal_vector(dir1)

scan2D_result = orqviz.scans.perform_2D_scan(params, loss_function, 
                                direction_x=dir1, direction_y=dir2,
                                n_steps_x=60)
orqviz.scans.plot_2D_scan_result(scan2D_result)
```
This code results in the following plot:
![Image](docs/example_plot.png)

## Further
TODO:
Troubleshooting\
API documentation ( that is normally required by public repositories)\
How to contribute guide\
How to get involved (slack,gitter etc.)\
Proper OSS license \
GitHub Issue templates\
Code of Conduct

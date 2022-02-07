![Image](docs/orqviz_logo.png)

# orqviz

A Python package for easily **visualizing the loss landscape** of Variational Quantum Algorithms by [Zapata Computing Inc](https://www.zapatacomputing.com/).

`orqviz` provides a collection of tools which quantum researchers and enthusiasts alike can use for their simulations. It works with any framework for running quantum circuits, for example `qiskit`, `cirq`, `pennylane`, and `Orquestra`. The package contains functions to generate data, as well as a range of flexible plotting and helper functions. `orqviz` is light-weight and has very few dependencies.

## Getting started

In [doc/examples/](https://github.com/zapatacomputing/orqviz/tree/main/docs/examples) we provide a range of `Jupyter notebook` examples for `orqviz`. We have four Jupyter notebooks with tutorials for how to get started with any quantum circuit simulation framework you might use. You will find examples with `qiskit`, `cirq`, `pennylane` and Zapata's `Orquestra` library. The tutorials are not exhaustive, but they do provide a full story that you can follow along.

In [this notebook](https://github.com/zapatacomputing/orqviz/blob/main/docs/examples/sombrero.ipynb) we have the _Sombrero_ example that we showcase in our paper. We also have an [advanced example notebook](https://github.com/zapatacomputing/orqviz/blob/main/docs/examples/advanced_example_notebook.ipynb) which provides a thorough demonstration of the flexibility of the `orqviz` package.

We recently published a paper on arXiv where we review the tools available with `orqviz`:\
[ORQVIZ: Visualizing High-Dimensional Landscapes in Variational Quantum Algorithms](https://arxiv.org/abs/2111.04695)

Find a brief overview of the visualization techniques on [YouTube](https://www.youtube.com/watch?v=_3x4NI6PcH4)!

## Installation

You can install our package using the following command:

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
    return np.sum(np.cos(pars))**2 + np.sum(np.sin(30*pars))**2

n_params = 42
params = np.random.uniform(-np.pi, np.pi, size=n_params)
dir1 = orqviz.geometric.get_random_normal_vector(n_params)
dir2 = orqviz.geometric.get_random_orthonormal_vector(dir1)

scan2D_result = orqviz.scans.perform_2D_scan(params, loss_function,
                                direction_x=dir1, direction_y=dir2,
                                n_steps_x=100)
orqviz.scans.plot_2D_scan_result(scan2D_result)
```

This code results in the following plot:

![Image](docs/example_plot.png)

## FAQ

**What are the expected type and shape for the parameters?**\
Parameters should be of type `numpy.ndarray` filled with real numbers. In recent releases, the shape of the parameters can be arbitrary, as long as `numpy` allows it, i.e., you cannot have inconsistent sizes per dimension. Until version `0.1.1`, the parameter array needed to be one-dimensional.

**What is the format of the `loss_function` that most `orqviz` methods expect?**\
We define a `loss_function` as a function which receives only the parameters of the model and returns a floating point/ real number. That value could for example be the cost function of an optimization problem, the prediction of a classifier, or the fidelity with respect to a fixed quantum state. All the calculation that needs to be performed to get to these values needs to happen in your function. Check out the above code as a minimal example.

**What can I do if my loss function requires additional arguments?**\
In that case you need to wrap the function into another function such that it again receives only the parameters of the model. We built a wrapper class called `LossFunctionWrapper` that you can import from `orqviz.loss_function`. It is a thin wrapper with helpful perks such as measuring the average evaluation time of a single loss function call, and the total number of calls.

## Authors

The leading developer of this package is Manuel Rudolph at Zapata Computing.\
For questions related to the visualization techniques, contact Manuel via manuel.rudolph@zapatacomputing.com .

The leading software developer of this package is Michał Stęchły at Zapata Computing.\
For questions related to technicalities of the package, contact Michał via michal.stechly@zapatacomputing.com .

Thank you to Sukin Sim and Luis Serrano from Zapata Computing for their contributions to the tutorials.

You can also contact us or ask general questions using [GitHub Discussions](https://github.com/zapatacomputing/orqviz/discussions).

For more specific code issues, bug fixes, etc. please open a [GitHub issue](https://github.com/zapatacomputing/orqviz/issues) in the `orqviz` repository.

If you are doing research using `orqviz`, please cite [our `orqviz` paper](https://arxiv.org/abs/2111.04695):

> Manuel S. Rudolph, Sukin Sim, Asad Raza, Michał Stęchły, Jarrod R. McClean, Eric R. Anschuetz, Luis Serrano, and Alejandro Perdomo-Ortiz. ORQVIZ: Visualizing High-Dimensional Landscapes in Variational Quantum Algorithms. 2021. arXiv:2111.04695

## How to contribute

Please see our [Contribution Guidelines](docs/CONTRIBUTING.md).

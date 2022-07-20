import os

import matplotlib.pyplot as plt
import numpy as np

from orqviz.hessians import (
    get_Hessian,
    perform_1D_hessian_eigenvector_scan,
    plot_1D_hessian_eigenvector_scan_result,
)
from orqviz.io import load_viz_object, save_viz_object


def COST_FUNCTION(params):
    return np.sum(np.sin(params)) + np.sum(params**2) + 3 * params[1] - 10 * params[2]


def test_plot_1D_hessian_eigenvector_scan_result():
    params = np.random.rand(8)

    hessian = get_Hessian(params, COST_FUNCTION, gradient_function=None, eps=1e-3)

    list_of_scans = perform_1D_hessian_eigenvector_scan(hessian, COST_FUNCTION)
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues)
    fig, ax = plt.subplots()
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues, ax=ax)

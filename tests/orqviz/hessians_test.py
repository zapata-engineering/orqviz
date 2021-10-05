import pytest
import numpy as np
import matplotlib.pyplot as plt
from zquantum.visualization.hessians import (
    get_Hessian,
    get_Hessian_SPSA_approx,
    HessianEigenobject,
    perform_1D_hessian_eigenvector_scan,
    plot_1D_hessian_eigenvector_scan_result,
)
from zquantum.visualization.utils import save_viz_object, load_viz_object


def COST_FUNCTION(params):
    return np.sum(np.sin(params)) + np.sum(params ** 2) + 3 * params[1] - 10 * params[2]


def test_get_hessian():
    params = np.random.rand(8)

    hessian = get_Hessian(params, COST_FUNCTION, gradient_function=None, eps=1e-3)

    assert hasattr(hessian, "eigenvectors")
    assert hasattr(hessian, "eigenvalues")
    assert (
        len(hessian.eigenvalues)
        == len(hessian.eigenvectors)
        == len(hessian.eigenvectors.T)
        == len(params)
    )

    list_of_scans = perform_1D_hessian_eigenvector_scan(hessian, COST_FUNCTION)
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues)
    fig, ax = plt.subplots()
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues, ax=ax)

    save_viz_object(hessian, "test")
    loaded_hessian = load_viz_object("test")
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvalues, hessian.eigenvalues
    )
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvectors, hessian.eigenvectors
    )


def test_get_hessian_SPSA_approx():
    params = np.random.rand(8)

    hessian = get_Hessian_SPSA_approx(
        params, COST_FUNCTION, gradient_function=None, eps=1e-3, n_reps=20
    )

    assert hasattr(hessian, "eigenvectors")
    assert hasattr(hessian, "eigenvalues")
    assert (
        len(hessian.eigenvalues)
        == len(hessian.eigenvectors)
        == len(hessian.eigenvectors.T)
        == len(params)
    )

    list_of_scans = perform_1D_hessian_eigenvector_scan(hessian, COST_FUNCTION)
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues)
    fig, ax = plt.subplots()
    plot_1D_hessian_eigenvector_scan_result(list_of_scans, hessian.eigenvalues, ax=ax)

    save_viz_object(hessian, "test")
    loaded_hessian = load_viz_object("test")
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvalues, hessian.eigenvalues
    )
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvectors, hessian.eigenvectors
    )

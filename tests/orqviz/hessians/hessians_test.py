import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orqviz.hessians import (
    get_Hessian,
    get_Hessian_SPSA_approx,
    perform_1D_hessian_eigenvector_scan,
    plot_1D_hessian_eigenvector_scan_result,
)
from orqviz.io import load_viz_object, save_viz_object


def COST_FUNCTION(params):
    return (
        np.sum(np.sin(params)) + np.sum(params**2) + params[0] ** 2 * params[1] ** 2
    )


def ANALYTICAL_HESSIAN(params):
    hessian = np.diag(2 - np.sin(params))
    hessian[0][0] += 2 * params[1] ** 2
    hessian[1][1] += 2 * params[0] ** 2
    hessian[0][1] = 4 * params[0] * params[1]
    hessian[1][0] = 4 * params[0] * params[1]
    return hessian


@pytest.mark.parametrize(
    "params",
    [np.random.rand(8)],
)
def test_get_hessian_io(params):
    hessian = get_Hessian(params, COST_FUNCTION, gradient_function=None, eps=1e-3)

    assert hasattr(hessian, "eigenvectors")
    assert hasattr(hessian, "eigenvalues")
    assert (
        len(hessian.eigenvalues)
        == len(hessian.eigenvectors)
        == len(hessian.eigenvectors.T)
        == len(params)
    )

    save_viz_object(hessian, "test")
    loaded_hessian = load_viz_object("test")
    os.remove("test")
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvalues, hessian.eigenvalues
    )
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvectors, hessian.eigenvectors
    )


@pytest.mark.parametrize(
    "params",
    [np.random.rand(8)],
)
def test_get_hessian_SPSA_approx_io(params):
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
    save_viz_object(hessian, "test")
    loaded_hessian = load_viz_object("test")
    os.remove("test")
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvalues, hessian.eigenvalues
    )
    np.testing.assert_array_almost_equal(
        loaded_hessian.eigenvectors, hessian.eigenvectors
    )


HESSIAN_TEST_PARAMS = [
    np.zeros(4),
    np.ones(5),
    np.arange(4, dtype=float),
    np.pi / 4 * np.arange(8),
]


@pytest.mark.parametrize("params", HESSIAN_TEST_PARAMS)
def test_get_hessian_gives_correct_values(params):
    eps = 1e-5
    target_matrix = ANALYTICAL_HESSIAN(params)
    target_eigenvalues = np.sort(np.linalg.eig(target_matrix)[0])

    hessian = get_Hessian(params, COST_FUNCTION, gradient_function=None, eps=eps)
    precision = int(np.abs(np.log10(eps))) - 1

    np.testing.assert_array_almost_equal(
        hessian.hessian_matrix, target_matrix, precision
    )
    np.testing.assert_array_almost_equal(
        hessian.eigenvalues, target_eigenvalues, precision
    )


@pytest.mark.parametrize("params", HESSIAN_TEST_PARAMS)
def test_get_hessian_spsa_gives_correct_values(params):
    eps = 1e-5
    target_matrix = ANALYTICAL_HESSIAN(params)
    target_eigenvalues = np.sort(np.linalg.eig(target_matrix)[0])

    hessian = get_Hessian_SPSA_approx(
        params, COST_FUNCTION, gradient_function=None, eps=eps, n_reps=20000
    )
    precision = 1
    np.testing.assert_array_almost_equal(
        hessian.hessian_matrix, target_matrix, precision
    )
    np.testing.assert_array_almost_equal(
        hessian.eigenvalues, target_eigenvalues, precision
    )

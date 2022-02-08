import numpy as np
import pytest

from orqviz.gradients import calculate_full_gradient


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def dSUM_OF_SINES_dPARAMS(params):
    return np.cos(params)


@pytest.mark.parametrize(
    "params",
    [
        np.array([1, 2, -3, 4, -1]),
        np.array([0.123, -0.987, 1, 6, 18, -23.8]),
        np.array([-np.pi, np.pi / 7, np.pi * 2]),
    ],
)
def test_numerical_gradient(params):

    gradient_values = calculate_full_gradient(
        params, SUM_OF_SINES, stochastic=False, eps=1e-6
    )
    assert gradient_values.shape == np.shape(params)
    assert np.allclose(gradient_values, dSUM_OF_SINES_dPARAMS(params))


@pytest.mark.parametrize(
    "params",
    [
        np.array([1, 2, -3, 4, -1]),
        np.array([0.123, -0.987, 1, 6, 18, -23.8]),
        np.array([-np.pi, np.pi / 7, np.pi * 2]),
    ],
)
def test_stochastic_gradient(params):

    gradient_values = np.mean(
        [
            calculate_full_gradient(
                params,
                SUM_OF_SINES,
                stochastic=True,
                eps=1e-6,
            )
            for _ in range(10000)
        ],
        axis=0,
    )
    assert gradient_values.shape == np.shape(params)
    assert np.allclose(gradient_values, dSUM_OF_SINES_dPARAMS(params), atol=0.1)

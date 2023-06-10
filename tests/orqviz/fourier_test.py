import numpy as np
import pytest
from matplotlib import pyplot as plt

import orqviz

np.random.seed(2)


def loss_function(pars, freq=3.5):
    return np.sum(np.cos(pars * freq))


n_params = 2
params = np.random.uniform(-np.pi, np.pi, size=n_params)
dir1 = np.array([1.0, 0.0])
dir2 = np.array([0.0, 1.0])
end_points = (0, 2 * np.pi)


# Make sure it works for even and odd resolutions
@pytest.mark.parametrize("res", [5, 6])
def test_fourier(res: int):
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=res,
        end_points_x=end_points,
    )

    assert fourier_result.values.shape == (res, res)
    assert fourier_result.end_points_x == fourier_result.end_points_y == end_points


@pytest.mark.parametrize("res", [5, 6])
def test_inverse(res: int):
    scan_2D_result = orqviz.scans.perform_2D_scan(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=6,  # the inverse of rfft always has an even resolution in the x dir
        n_steps_y=res,
        end_points_x=end_points,
    )
    transformed = orqviz.fourier.perform_2D_fourier_transform(
        scan_2D_result, end_points, end_points
    )
    inversed = orqviz.fourier.inverse_fourier(transformed)

    np.testing.assert_array_almost_equal(scan_2D_result.values, inversed.values)


def test_plots_dont_fail():
    res = 10
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=res,
        end_points_x=end_points,
    )
    orqviz.fourier.plot_2D_fourier_result(fourier_result, max_freq_x=3, max_freq_y=3)
    orqviz.fourier.plot_2D_fourier_result(
        fourier_result, max_freq_x=3, max_freq_y=3, show_full_spectrum=True
    )
    fig, ax = plt.subplots()
    orqviz.fourier.plot_2D_fourier_result(
        fourier_result, max_freq_x=3, max_freq_y=3, fig=fig, ax=ax
    )


def test_inverse_plots_dont_fail():
    res = 5
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=res,
        end_points_x=end_points,
    )
    inverse_result = orqviz.fourier.inverse_fourier(fourier_result)
    orqviz.fourier.plot_inverse_fourier_result(inverse_result)
    fig, ax = plt.subplots()
    orqviz.fourier.plot_inverse_fourier_result(fourier_result, fig, ax)

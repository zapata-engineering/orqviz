import numpy as np
import pytest
from matplotlib import pyplot as plt

import orqviz
from orqviz.fourier import (
    _move_negative_frequencies_next_to_origin,
    _move_negative_frequencies_next_to_positive_frequencies,
    _truncate_result_according_to_resolution,
)

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
    expected_y_res = res
    expected_x_res = res // 2 + 1  # Because rfft is symmetric
    assert fourier_result.values.shape == (expected_y_res, expected_x_res)
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


@pytest.mark.parametrize("input_res", [6, 7])
def test_truncate(input_res: int):
    max_freq = 2
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=input_res,
        end_points_x=end_points,
    )

    truncated_result = _truncate_result_according_to_resolution(
        fourier_result.values, max_freq, max_freq
    )

    # Make sure output resolution is correct
    expected_y_res = max_freq * 2 + 1
    expected_x_res = max_freq + 1
    assert truncated_result.shape == (expected_y_res, expected_x_res)


def test_helper_swap_functions_are_inverses():
    for dim in range(1, 6):
        arbitrary_arr = np.random.rand(dim, np.random.randint(1, 5))
        np.testing.assert_allclose(
            _move_negative_frequencies_next_to_positive_frequencies(
                _move_negative_frequencies_next_to_origin(arbitrary_arr)
            ),
            arbitrary_arr,
        )


def test_plots():
    res = 5
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=res,
        end_points_x=end_points,
    )
    orqviz.fourier.plot_2D_fourier_result(fourier_result)
    orqviz.fourier.plot_2D_fourier_result(
        fourier_result, show_negative_frequencies=True
    )
    fig, ax = plt.subplots()
    orqviz.fourier.plot_2D_fourier_result(fourier_result, fig=fig, ax=ax)


@pytest.mark.parametrize(
    # If resolution is 5, the frequencies of -2, -1, 0, 1, and 2 will be included
    # (before normalization). So setting max_freq to 3 when end points are 0-2pi should
    # cause a warning.
    # When end points are 0-4pi, a resolution of 5 means that the frequencies of -1,
    # -0.5, 0, 0.5, and 1 are included. So setting max_freq to 2 should cause a warning.
    "max_freq,end_points",
    [(3, (0, 2 * np.pi)), (2, (0, 4 * np.pi))],
)
def test_plotting_raises_warning_if_max_freq_exceeds_normalized_resolution(
    max_freq, end_points
):
    res = 5
    fourier_result = orqviz.fourier.scan_2D_fourier(
        params,
        loss_function,
        direction_x=dir1,
        direction_y=dir2,
        n_steps_x=res,
        end_points_x=end_points,
    )
    with pytest.warns(UserWarning):
        orqviz.fourier.plot_2D_fourier_result(
            fourier_result, max_freq_x=max_freq, max_freq_y=max_freq
        )


def test_inverse_plots():
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

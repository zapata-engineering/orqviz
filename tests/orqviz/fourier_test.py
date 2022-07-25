import orqviz
import numpy as np
from orqviz.fourier import _iswap, _swap, _truncate_result_according_to_resolution
import pytest


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


def test_swap_and_iswap_are_inverses():
    for dim in range(1, 6):
        arbitrary_arr = np.random.rand(dim, np.random.randint(1, 5))
        np.testing.assert_allclose(_iswap(_swap(arbitrary_arr)), arbitrary_arr)

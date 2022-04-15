from .scans.scans_2D import perform_2D_scan
import numpy as np
from typing import Optional, Tuple

import numpy as np

from .aliases import (
    DirectionVector,
    LossFunction,
    ParameterVector,
)
from .scans.data_structures import Scan2DResult
import warnings


def scan_2D_dct(
    origin: ParameterVector,
    loss_function: LossFunction,
    direction_x: Optional[DirectionVector] = None,
    direction_y: Optional[DirectionVector] = None,
    n_steps_x: int = 20,
    n_steps_y: Optional[int] = None,
    end_points_x: Tuple[float, float] = (-1, 1),
    end_points_y: Optional[Tuple[float, float]] = None,
    dct_resolution_x: int = None,
    dct_resolution_y: int = None,
):
    if n_steps_y is None:
        n_steps_y = n_steps_x
    if dct_resolution_x is None:
        dct_resolution_x = n_steps_x
    if dct_resolution_y is None:
        dct_resolution_y = dct_resolution_x
    if end_points_y is None:
        end_points_y = end_points_x
    scan2D_result = perform_2D_scan(
        origin,
        loss_function,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
    )
    result = dct_2D(scan2D_result.values)

    frequency_normalization_factor_x = (end_points_x[1] - end_points_x[0]) / np.pi
    frequency_normalization_factor_y = (end_points_y[1] - end_points_y[0]) / np.pi
    normalized_x_res = dct_resolution_x * int(frequency_normalization_factor_x)
    normalized_y_res = dct_resolution_y * int(frequency_normalization_factor_y)
    if normalized_x_res > n_steps_x:
        warnings.warn(
            "DCT resolution X is too low for the number of steps so the default resolution will be used."
        )
        dct_resolution_x = n_steps_x

    if normalized_y_res > n_steps_y:
        warnings.warn(
            "DCT resolution Y is too low for the number of steps so the default resolution will be used."
        )
        dct_resolution_y = n_steps_y

    grid = np.array(
        [
            [
                [
                    x / frequency_normalization_factor_x,
                    y / frequency_normalization_factor_y,
                ]
                for x in range(normalized_x_res)
            ]
            for y in range(normalized_y_res)
        ]
    )
    # Truncate result according with resolution.
    result = result[0:normalized_y_res, 0:normalized_x_res]
    dir1 = np.array([1.0, 0.0])
    dir2 = np.array([0.0, 1.0])
    return Scan2DResult(grid, dir1, dir2, result)


def dct_2D(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    intermediate = []
    for row in x:
        intermediate.append(dct(row))
    intermediate = np.array(intermediate).T
    result = []
    for row in intermediate:
        result.append(dct(row))
    return np.array(result).T


def dct(x: np.ndarray) -> np.ndarray:
    """DCT-II 1D transformation. It's normalized."""
    assert len(x.shape) == 1
    N = x.size
    X = np.zeros(N)
    for k in range(N):
        X[k] = np.dot(x, np.cos((np.arange(N) + 0.5) * np.pi / N * k))
    # Normalize result by factor of X * 2 / N
    return X * 2 / N

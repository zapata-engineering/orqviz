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
    # should x and y even be diff arguments? like should we support dct on non-squares?
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
    result = dct_2d(scan2D_result.values)
    if n_steps_y is None:
        n_steps_y = n_steps_x
    if dct_resolution_x is None:
        dct_resolution_x = n_steps_x
    if dct_resolution_y is None:
        dct_resolution_y = dct_resolution_x
    frequency_normalization_factor = (end_points_x[1] - end_points_x[0]) / np.pi
    normalized_x_res = dct_resolution_x * int(frequency_normalization_factor)
    normalized_y_res = dct_resolution_y * int(frequency_normalization_factor)
    grid = np.array(
        [
            [
                [x / frequency_normalization_factor, y / frequency_normalization_factor]
                for x in range(normalized_x_res)
            ]
            for y in range(normalized_y_res)
        ]
    )
    # Truncate result according with resolution.
    result = result[0:normalized_x_res, 0:normalized_y_res]
    return Scan2DResult(grid, direction_x, direction_y, result)


def dct_2d(x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2
    intermediate = []
    for row in x:
        intermediate.append(dct(row))
    intermediate = np.array(intermediate)
    intermediate = intermediate.T
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

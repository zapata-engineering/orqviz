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
    intermediate = []
    for row in scan2D_result.values:
        intermediate.append(dct(row))
    intermediate = np.array(intermediate)
    intermediate = intermediate.T
    result = []
    for row in intermediate:
        result.append(dct(row))
    result = np.array(result).T
    if n_steps_y is None:
        n_steps_y = n_steps_x
    if dct_resolution_x is None:
        dct_resolution_x = n_steps_x
    if dct_resolution_y is None:
        dct_resolution_y = dct_resolution_x
    grid = np.array(
        [[[x, y] for x in range(dct_resolution_x)] for y in range(dct_resolution_y)]
    )
    # truncate result according with resolution.
    result = result[0:dct_resolution_x, 0:dct_resolution_y]
    return Scan2DResult(grid, direction_x, direction_y, result)


def dct(x: np.ndarray) -> np.ndarray:
    """DCT-II"""
    assert len(x.shape) == 1
    N = x.size
    X = np.zeros(N)
    for k in range(N):
        X[k] = np.dot(x, np.cos((np.arange(N) + 0.5) * np.pi / N * k))
    return X


def dct_1(x: np.ndarray) -> np.ndarray:
    """DCT-I"""
    assert len(x.shape) == 1
    N = x.size
    X = np.zeros(N)
    for k in range(N):
        term_1 = 0.5 * (x[0] + (-1) ** k * x[N - 1])
        # term_2 = 0
        # for n in range(1, N - 1):
        #     term_2 += x[n] * np.cos((np.pi / (N - 1)) * n * k)
        term_2 = np.dot(x[1:-1], np.cos((np.pi / (N - 1)) * (np.arange(N - 2) + 1) * k))
        X[k] = term_1 + term_2
    return X

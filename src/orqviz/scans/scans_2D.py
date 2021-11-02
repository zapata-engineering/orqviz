from typing import Callable, Optional, Tuple

import numpy as np

from ..aliases import GridOfParameterVectors, ParameterVector
from ..geometric import (
    get_random_normal_vector,
    get_random_orthonormal_vector,
    relative_periodic_wrap,
)
from .data_structures import Scan2DResult
from .evals import eval_points_on_grid


def perform_2D_interpolation(
    point_1: ParameterVector,
    point_2: ParameterVector,
    loss_function: Callable[[ParameterVector], float],
    direction_y: Optional[np.ndarray] = None,
    n_steps_x: int = 20,
    n_steps_y: Optional[int] = None,
    end_points_x: Tuple[float, float] = (-0.5, 1.5),
    end_points_y: Tuple[float, float] = (-0.5, 0.5),
    parameter_period: Optional[float] = None,
    verbose: bool = False,
) -> Scan2DResult:
    """Function to perform a 2D scan to interpolate between two points.

    Args:
        point_1: First point of the interpolation.
        point_2: Second point of the interpolation.
        loss_function: Loss function to scan.
        direction_y: Second scan direction for the 2D scan where first direction
            is the interpolation vector of the points. If None, it's chosen at random.
        n_steps_x: Number of points evaluated along the x-direction. Defaults to 20.
        n_steps_y: Number of points evaluated along the y-direction.
            If None, set value to n_steps_y. Defaults to None.
        end_points_x: Range of scan along the x-direction in units
            of the interpolation vector. Defaults to (-0.5, 1.5).
        end_points_y: Range of scan along the y-direction in units
            of the interpolation vector. Defaults to (-0.5, 0.5).
        parameter_period (Optional[float], optional): Optional period of
        the parameters to scan the shortest interpolated path between the points.
            If None, interpolation per parameter happens along the real number line.
            Defaults to None.
        verbose: Flag for printing progress. Defaults to False.
    """
    if n_steps_y is None:
        n_steps_y = n_steps_x

    if parameter_period:
        point_2 = relative_periodic_wrap(point_1, point_2, parameter_period)

    direction_x = point_2 - point_1

    if direction_y is None:
        direction_y = get_random_orthonormal_vector(direction_x) * np.linalg.norm(
            direction_x
        )

    return perform_2D_scan(
        origin=point_1,
        loss_function=loss_function,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
        verbose=verbose,
    )


def perform_2D_scan(
    origin: ParameterVector,
    loss_function: Callable[[ParameterVector], float],
    direction_x: Optional[np.ndarray] = None,
    direction_y: Optional[np.ndarray] = None,
    n_steps_x: int = 20,
    n_steps_y: Optional[int] = None,
    end_points_x: Tuple[float, float] = (-1, 1),
    end_points_y: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
) -> Scan2DResult:
    """Function to perform a 2D scan around a point on a loss function.

    Args:
        origin: Origin point of the 2D scan.
        loss_function: Loss function to be scanned.
        direction_x: x-direction vector for scan. Has same shape as origin.
            If None, a random unit vector is sampled. Defaults to None.
        direction_y: y-direction vector for scan. Has same shape as origin.
            If None, a random unit vector is sampled. Defaults to None.
        n_steps_x: Number of points evaluated along the x-direction. Defaults to 20.
        n_steps_y: Number of points evaluated along the y-direction.
            If None, set value to n_steps_y. Defaults to None.
        end_points_x: Range of scan along the x-direction in units of direction_x.
            Defaults to (-1, 1).
        end_points_y: Range of scan along the x-direction in units of direction_x.
            Defaults to (-1, 1).
        verbose: Flag for printing progress. Defaults to False.
    """
    if direction_x is None:
        direction_x = get_random_normal_vector(len(origin))
    if direction_y is None:
        direction_y = get_random_orthonormal_vector(direction_x) * np.linalg.norm(
            direction_x
        )
    if n_steps_y is None:
        n_steps_y = n_steps_x

    if end_points_y is None:
        end_points_y = end_points_x

    interpolation_steps_x = np.linspace(
        end_points_x[0], end_points_x[1], num=n_steps_x, endpoint=True
    )
    interpolation_steps_y = np.linspace(
        end_points_y[0], end_points_y[1], num=n_steps_y, endpoint=True
    )
    x_values, y_values = np.meshgrid(interpolation_steps_x, interpolation_steps_y)

    # We acknowledge the fact that this method is not the most intuitive one,
    # but it does the job, so we decided to leave it here.
    params_grid: GridOfParameterVectors = (
        origin
        + np.dot(x_values[:, :, None], direction_x[None, :])
        + np.dot(y_values[:, :, None], direction_y[None, :])
    )

    loss_grid = eval_points_on_grid(
        params_grid, loss_function=loss_function, n_reps=1, verbose=verbose
    )
    return Scan2DResult(
        params_grid=params_grid,
        direction_x=direction_x,
        direction_y=direction_y,
        values=loss_grid,
        origin=origin,
    )

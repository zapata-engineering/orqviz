from typing import Callable, Optional, Tuple

import numpy as np

from ..aliases import ArrayOfParameterVectors, ParameterVector
from ..geometric import (
    direction_linspace,
    get_random_normal_vector,
    relative_periodic_wrap,
)
from .data_structures import Scan1DResult
from .evals import eval_points_on_path


def perform_1D_scan(
    origin: ParameterVector,
    loss_function: Callable[[ParameterVector], float],
    direction: Optional[np.ndarray] = None,
    n_steps: int = 31,
    end_points: Tuple[float, float] = (-np.pi, np.pi),
    verbose: bool = False,
) -> Scan1DResult:
    """Function to perform a 1D scan on a loss function in a specific direction.

    Args:
        origin: Parameter vector that is the origin on the 1D scan.
        loss_function: Loss function to perform the scan on.
        direction: Direction in which loss function is scanned around the origin.
            If None, a random unit vector is sampled. Defaults to None
        n_points: Number of points to evaluate along the scan. Defaults to 31.
        end_points: Range of scan along the direction in units of the direction vector.
            Defaults to (-np.pi, np.pi).
        verbose: Flag for printing progress. Defaults to False.
    """
    if direction is None:
        direction = get_random_normal_vector(len(origin))

    point_list: ArrayOfParameterVectors = direction_linspace(
        origin=origin,
        direction=direction,
        n_points=n_steps,
        endpoints=end_points,
    )

    scan_values = eval_points_on_path(point_list, loss_function, verbose=verbose)
    return Scan1DResult(
        point_list, direction=direction, values=scan_values, origin=origin
    )


def perform_1D_interpolation(
    point_1: ParameterVector,
    point_2: ParameterVector,
    loss_function: Callable[[ParameterVector], float],
    n_steps: int = 100,
    end_points: Tuple[float, float] = (-0.5, 1.5),
    parameter_period: Optional[float] = None,
    verbose: bool = False,
) -> Scan1DResult:
    """Function to perform a 1D scan to interpolate between two points.

    Args:
        point_1: First point of the interpolation.
        point_2: Second point of the interpolation.
        loss_function: Loss function to scan.
        n_steps: Number of points evaluated along the scan. Defaults to 100.
        end_points: Range of scan along the direction in units of the
            interpolation vector. Defaults to (-0.5, 1.5).
        parameter_period: Optional period of the parameters to scan the shortest
            interpolated path between the points.
            If None, interpolation per parameter happens along the real number line.
            Defaults to None.
        verbose: Flag for printing progress. Defaults to False.
    """
    if parameter_period:
        point_2 = relative_periodic_wrap(point_1, point_2, parameter_period)

    origin = point_1
    direction = point_2 - point_1
    point_list: ArrayOfParameterVectors = direction_linspace(
        origin=origin,
        direction=direction,
        n_points=n_steps,
        endpoints=end_points,
    )

    loss_vector = eval_points_on_path(
        point_list, loss_function=loss_function, verbose=verbose
    )

    return Scan1DResult(
        point_list,
        direction=direction,
        values=loss_vector,
        origin=origin,
    )

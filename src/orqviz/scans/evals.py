from typing import Callable, List, Optional

import numpy as np

from ..aliases import ArrayOfParameterVectors, GridOfParameterVectors, ParameterVector


def eval_points_on_path(
    all_points: ArrayOfParameterVectors,
    loss_function: Callable[[ParameterVector], float],
    n_reps: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Function to evaluate loss function on a 1D path of parameters.

    Args:
        all_parameters: Array of parameters with shape (len, n_params)
        loss_function: Loss function to evaluate the parameter array
        n_reps: Repetitions to average the output in noisy cases. Defaults to 1.
        verbose: Flag for verbosity of progress. Defaults to False.

    """
    n_points = len(all_points)

    values: List[List[Optional[float]]] = [[None] * n_points] * n_reps
    for rep in range(n_reps):
        for idx, point in enumerate(all_points):
            if idx % 10 == 0 and verbose:
                print("Progress: {:.1f}%".format(round(idx / n_points * 100)))
            values[rep][idx] = loss_function(point)

    return np.mean(np.asarray(values), axis=0)


def eval_points_on_grid(
    all_parameters: GridOfParameterVectors,
    loss_function: Callable[[ParameterVector], float],
    n_reps: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Function to evaluate loss function on a 2D grid of parameters.

    Args:
        all_parameters: Grid of parameters with shape (len_y, len_x, n_params)
        loss_function: Loss function to evaluate the parameter grid
        n_reps: Repetitions to average the output in noisy cases. Defaults to 1.
        verbose: Flag for verbosity of progress. Defaults to False.

    """

    size_x, size_y, n_params = np.array(all_parameters).shape

    vector_of_parameters = all_parameters.reshape((size_x * size_y, n_params))
    vector_of_values = eval_points_on_path(
        all_points=vector_of_parameters,
        loss_function=loss_function,
        n_reps=n_reps,
        verbose=verbose,
    )
    return vector_of_values.reshape((size_x, size_y))

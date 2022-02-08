from typing import Callable, List, Optional

import numpy as np

from ..aliases import (
    ArrayOfParameterVectors,
    GridOfParameterVectors,
    LossFunction,
    ParameterVector,
)


def eval_points_on_path(
    all_points: ArrayOfParameterVectors,
    loss_function: LossFunction,
    n_reps: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Function to evaluate loss function on a 1D path of parameters.

    Args:
        all_parameters: Array of parameters with shape (len, *(parameters.shape))
        loss_function: Function to evaluate the parameters on. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
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

    return np.array(np.mean(np.asarray(values), axis=0))


def eval_points_on_grid(
    all_parameters: GridOfParameterVectors,
    loss_function: LossFunction,
    n_reps: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Function to evaluate loss function on a 2D grid of parameters.

    Args:
        all_parameters:
            Grid of parameters with shape (len_y, len_x, *(parameters.shape))
        loss_function: Function toevaluate the parameters on. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        n_reps: Repetitions to average the output in noisy cases. Defaults to 1.
        verbose: Flag for verbosity of progress. Defaults to False.

    """

    shape = np.shape(all_parameters)
    (size_x, size_y), params_shape = shape[:2], shape[2:]

    vector_of_parameters = all_parameters.reshape((size_x * size_y, *params_shape))

    vector_of_values = eval_points_on_path(
        all_points=vector_of_parameters,
        loss_function=loss_function,
        n_reps=n_reps,
        verbose=verbose,
    )
    return vector_of_values.reshape((size_x, size_y))

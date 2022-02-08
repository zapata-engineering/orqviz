from typing import Callable, Optional, Tuple

import numpy as np

from orqviz.aliases import (
    ArrayOfParameterVectors,
    FullGradientFunction,
    LossFunction,
    ParameterVector,
)
from orqviz.gradients import calculate_full_gradient


def gradient_descent_optimizer(
    init_params: ParameterVector,
    loss_function: LossFunction,
    n_iters: int,
    learning_rate: float = 0.1,
    full_gradient_function: FullGradientFunction = None,
    eval_loss_during_training: bool = True,
) -> Tuple[ArrayOfParameterVectors, np.ndarray]:
    """Function perform gradient descent optimization on a loss function.

    Args:
        init_params: Initial parameter vector from which to start the optimization.
        loss_function: Function with respect to which the gradient is calculated.
            It must receive only a numpy.ndarray of parameters, and return
            a real number. If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        n_iters: Number of iterations to optimize.
        learning_rate: Learning rate for gradient descent. The calculated gradient
            is multiplied with this value and then updates the parameter vector.
        full_gradient_function: Gradient function to calculate the partial derivatives
            of the loss function with respect to each parameter vector entry.
            If None, a simple numerical gradient is computed. Defaults to None.
        eval_loss_during_training: Flag to indicate whether to evaluate the loss
            at every iteration during training. Doing so adds 'n_iters-1'
            loss function calls. If False, only the initial and final losses
            are calculated and returned. Defaults to False.
    """
    if full_gradient_function is None:

        def _full_gradient_function(params):
            return calculate_full_gradient(params, loss_function=loss_function)

        full_gradient_function = _full_gradient_function

    params = init_params
    all_costs = []
    all_params = [init_params]
    for _ in range(n_iters):
        if eval_loss_during_training:
            all_costs.append(loss_function(params))
        grad_val = full_gradient_function(params)
        params = params - learning_rate * grad_val
        all_params.append(params)

    all_costs.append(loss_function(params))

    return np.array(all_params), np.array(all_costs)

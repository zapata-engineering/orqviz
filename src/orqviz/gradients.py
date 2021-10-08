from typing import Callable, Optional

import numpy as np

from .aliases import ParameterVector


def calculate_full_gradient(
    params: ParameterVector,
    loss_function: Callable[[ParameterVector], float],
    gradient_function: Optional[Callable[[ParameterVector, np.ndarray], float]] = None,
    stochastic: bool = False,
    eps: float = 0.1,
) -> ParameterVector:
    """Function to calculate a full gradient vector of partial derivatives
        of a loss function with respect to each entry of a parameter vector.

    Args:
        params: Parameter vector at which to calculate the gradient.
        loss_function: Loss function with respect to which the gradient is calculated.
            It is not required when a gradient_function is passed.
        gradient_function: Gradient function to calculate the partial derivative
            of the loss function with respect to a direction.
            If None, a numerical gradient is computed with the passed loss function.
            Defaults to None.
        stochastic: Flag to indicate whether a stochastic gradient is computed in
            a randomly sampled direction.
            If True, the specified gradient function must support this.
            Defaults to False.
        eps: Stencil for numerical gradient calculation. Defaults to 0.1.
    """

    if loss_function is None and gradient_function is None:
        raise Exception(
            """
            'loss_function' and 'gradient_function' cannot both be 'None'
            """
        )

    if gradient_function is None:

        def _gradient_function(pars: ParameterVector, direction: np.ndarray) -> float:
            return numerical_gradient(
                pars, direction, loss_function=loss_function, eps=eps
            )

        gradient_function = _gradient_function

    grad = np.zeros_like(params)

    if stochastic:
        direction = np.random.choice([0.0, 1.0], size=np.shape(params))
        grad_value = gradient_function(params, direction)
        grad += grad_value * direction
    else:
        for i in range(len(params)):
            direction = np.zeros_like(params)
            direction[i] = 1.0
            grad_value = gradient_function(params, direction)
            grad += grad_value * direction
    return grad


def numerical_gradient(
    x: ParameterVector,
    direction: np.ndarray,
    loss_function: Callable[[ParameterVector], float],
    eps: float,
) -> float:
    """Function to calculate a numerical gradient of a loss function at point x
    with respect to a specified direction."""
    x_plus = x + eps * direction
    f_plus = loss_function(x_plus)

    x_minus = x - eps * direction
    f_minus = loss_function(x_minus)
    return (f_plus - f_minus) / (2 * eps)

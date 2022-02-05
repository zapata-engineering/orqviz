from typing import Callable, Optional

import numpy as np

from .aliases import DirectionVector, GradientFunction, LossFunction, ParameterVector


def calculate_full_gradient(
    params: ParameterVector,
    loss_function: LossFunction,
    gradient_function: GradientFunction = None,
    stochastic: bool = False,
    eps: float = 1e-3,
) -> ParameterVector:
    """Function to calculate a full gradient vector of partial derivatives
        of a loss function with respect to each entry of a parameter vector.

    Args:
        params: Parameter vector at which to calculate the gradient.
        loss_function: Function with respect to which the gradient is calculated.
            It is not required when a gradient_function is passed. It must receive
            only a numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        gradient_function: Gradient function to calculate the partial derivative
            of the loss function with respect to a direction.
            If None, a numerical gradient is computed with the passed loss function.
            Defaults to None.
        stochastic: Flag to indicate whether a stochastic gradient is computed in
            a randomly sampled direction.
            If True, the specified gradient function must support this.
            Defaults to False.
        eps: Stencil for numerical gradient calculation. If the loss function is noisy,
            this value needs to be increase, e.g to 0.1. Defaults to 1e-3.
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
        direction = np.random.choice([-1.0, 1.0], size=np.shape(params))
        grad_value = gradient_function(params, direction)
        grad = grad + grad_value * direction
    else:
        flat_params = params.flatten()
        for i in range(len(flat_params)):
            direction = np.zeros_like(flat_params)
            direction[i] = 1.0
            direction = direction.reshape(params.shape)
            grad_value = gradient_function(params, direction)
            grad = grad + grad_value * direction
    return grad


def numerical_gradient(
    x: ParameterVector,
    direction: DirectionVector,
    loss_function: LossFunction,
    eps: float,
) -> float:
    """Function to calculate a numerical gradient of a loss function at point x
    with respect to a specified direction."""
    x_plus = x + eps * direction
    f_plus = loss_function(x_plus)

    x_minus = x - eps * direction
    f_minus = loss_function(x_minus)
    return (f_plus - f_minus) / (2 * eps)

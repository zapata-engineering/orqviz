from typing import Callable, List, Optional, Tuple

import numpy as np

from ..aliases import GradientFunction, LossFunction, ParameterVector
from ..gradients import numerical_gradient
from ..scans.data_structures import Scan1DResult
from ..scans.scans_1D import perform_1D_scan
from .data_structures import HessianEigenobject


def perform_1D_hessian_eigenvector_scan(
    hessian_object: HessianEigenobject,
    loss_function: LossFunction,
    n_points: int = 31,
    endpoints: Tuple[float, float] = (-np.pi, np.pi),
) -> List[Scan1DResult]:
    """Function to create 1D scans along all eigenvecor directions
        of a Hessian and return a list of scans.

    Args:
        hessian_object: HessianEigenobject Datatype containing a Hessian matrix.
        loss_function: Function to perform the scan on. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        n_points: Number of points to evaluate the loss along each direction.
            Defaults to 31.
        endpoints: End points for scan along each direction.
            Defaults to (-np.pi, np.pi).
    """
    list_of_scans = []
    for direction in hessian_object.eigenvectors:
        scan = perform_1D_scan(
            hessian_object.params, loss_function, direction, n_points, endpoints
        )
        list_of_scans.append(scan)

    return list_of_scans


def get_Hessian(
    params: ParameterVector,
    loss_function: LossFunction,
    gradient_function: Optional[GradientFunction] = None,
    n_reps: int = 1,
    eps: float = 0.1,
) -> HessianEigenobject:
    """Function to calculate the Hessian matrix of a loss function
        (matrix of second-order partial derivatives) at a specified parameter vector.
    The calculation is performed by combining an optional gradient function
        and finite difference gradients.
    Calculation time scales quadratically with the number of parameters.

    Args:
        params: Parameter vector at which the Hessian matrix is computed.
        loss_function: Function to calculate the Hessian of. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        gradient_function: Gradient function which can be used to calculate
            the partial derivative of the loss function for individial parameters.
            It can be used to avoid some numerical gradients and improve
            accuracy of the Hessian. Defaults to None.
        n_reps: Number of repetitions over which Hessian is averaged.
            Useful if noise is present. Defaults to 1.
        eps: Finite difference stencil used for numerical gradient in calculation.
            It is always used, even if gradient function is provided. Defaults to 0.1.
    """

    flat_params = params.flatten()

    n_params = len(flat_params)
    hessian_shape = (n_params, n_params)
    Hessian_Matr = np.zeros(shape=hessian_shape)

    if gradient_function is None:

        def _gradient_function(x: ParameterVector, d: ParameterVector) -> float:
            return numerical_gradient(x, d, loss_function, eps)

        gradient_function = _gradient_function

    for _ in range(n_reps):
        for j in range(n_params):
            dir1 = np.zeros_like(flat_params)
            dir1[j] = 1
            dir1 = dir1.reshape(params.shape)
            dir1_gradient = gradient_function(params, dir1)
            for k in range(0, j + 1):
                dir2 = np.zeros_like(flat_params)
                dir2[k] = 1
                dir2 = dir2.reshape(params.shape)
                dir2_gradient = gradient_function(params + dir2 * eps, dir1)

                op = np.outer(dir1.flatten(), dir2.flatten())
                outer_prod_matrix = op + op.T

                hessian_result = (dir2_gradient - dir1_gradient) / eps
                Hessian_Matr += hessian_result * outer_prod_matrix

    return HessianEigenobject(params, Hessian_Matr / n_reps)


def get_Hessian_SPSA_approx(
    params: ParameterVector,
    loss_function: LossFunction,
    gradient_function: Optional[GradientFunction] = None,
    n_reps: int = 20,
    eps: float = 0.1,
) -> HessianEigenobject:
    """Function to calculate an SPSA approximation of the the Hessian matrix of
        a loss function (matrix of second-order partial derivatives)
        at a specified parameter vector.
    The Hessian is approximated by estimating the second partial derivative
        in random stochastic directions.

    Args:
        params: Parameter vector at which the Hessian matrix is computed.
        loss_function: Function to calculate the Hessian of. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        gradient_function: Gradient function which can be used to calculate
            the derivative of the loss function in random stochastic directions.
            It can be used to avoid some numerical gradients and improve accuracy
            of the Hessian. Defaults to None.
        n_reps: Number of random stochastic gradients used to approximate the Hessian.
            Defaults to 20.
        eps: Finite difference stencil used for numerical gradient in calculation.
            It is always used, even if gradient function is provided. Defaults to 0.1.
    """

    flat_params = params.flatten()

    n_params = len(flat_params)
    hessian_shape = (n_params, n_params)
    Hessian_Matr = np.zeros(shape=hessian_shape)

    if gradient_function is None:

        def _gradient_function(x: ParameterVector, d: ParameterVector) -> float:
            return numerical_gradient(x, d, loss_function, eps)

        gradient_function = _gradient_function

    for _ in range(n_reps):
        dir1 = np.random.choice([-1.0, 1.0], size=params.shape)
        dir1_gradient = gradient_function(params, dir1)

        dir2 = np.random.choice([-1.0, 1.0], size=params.shape)
        dir2_gradient = gradient_function(params + dir2 * eps, dir1)

        op = np.outer(dir1.flatten(), dir2.flatten())
        outer_prod_matrix = (op + op.T) / 2

        hessian_result = (dir2_gradient - dir1_gradient) / eps
        Hessian_Matr += hessian_result * outer_prod_matrix

    return HessianEigenobject(params, Hessian_Matr / n_reps)

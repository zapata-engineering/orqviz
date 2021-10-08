from typing import Callable, List, Optional

import numpy as np

from ..aliases import ParameterVector, Weights
from ..gradients import calculate_full_gradient
from .data_structures import Chain, ChainPath


def run_NEB(
    init_chain: Chain,
    loss_function: Callable[[ParameterVector], float],
    full_gradient_function: Optional[Callable[[ParameterVector], np.ndarray]] = None,
    n_iters: int = 10,
    eps: float = 0.1,
    learning_rate: float = 0.1,
    stochastic: bool = False,
    calibrate_tangential: bool = False,
    cummulative_weights: Optional[Weights] = None,
    verbose: bool = False,
) -> List[Chain]:
    """Nudged Elastic Band (NEB) algorithm to train a piece-wise linear path optimized
        in the loss function landscape.
    Reference paper:
        Essentially No Barriers in Neural Network Energy Landscape, arXiv:1803.00885.
    NOTE: There is a discrepancy between name used in the original paper (elastic band)
        and the one we use to define our data structure (chain), but we decided to use
        the later as we feel it's more appropriate for this data structure.

    Args:
        init_chain: Initial chain that is optimized with the algorithm.
        loss_function: Loss function that is used to optimize the chain.
        full_gradient_function: Function to calculate the gradient w.r.t.
            the loss function for all parameters. Defaults to None.
        n_iters: Number of optimization iterations. Defaults to 10.
        eps: Stencil for finite difference gradient if full_gradient_function
            is not provided. Defaults to 0.1.
        learning_rate: Learning rate/ step size for the gradient descent optimization.
            Defaults to 0.1.
        stochastic: Flag to indicate whether to perform stochastic gradient descent
            if full_gradient_function is not provided.
            It is less stable but much faster. Defaults to False.
        calibrate_tangential: Flag to indicate whether next neighbor for finding
            tangential direction is calibrated with an additional loss evaluation.
            Defaults to False.
        cummulative_weights: Cummulative chain position weights to re-distributed
            pivots along chain. If None, pivots are re-distributed uniformly.
            Defaults to None.
        verbose: Flag for printing progress. Defaults to False.

    Returns:
        All chains during training
    """
    if full_gradient_function is None:
        # Defines an automatic numerical gradient with eps, stochastic as args
        def _full_gradient_function(pars: ParameterVector) -> ParameterVector:
            return calculate_full_gradient(
                pars, loss_function=loss_function, stochastic=stochastic, eps=eps
            )

        full_gradient_function = _full_gradient_function

    all_chains = [init_chain]
    chain = init_chain
    for it in range(n_iters):
        if it % 5 == 0 and verbose:
            print("iteration", it)
        gradients_on_pivots = _get_gradients_on_pivots(
            chain,
            loss_function,
            full_gradient_function,
            calibrate_tangential,
        )
        new_pivots = np.asarray(
            [
                old_pivot - learning_rate * gradient
                for old_pivot, gradient in zip(chain.pivots, gradients_on_pivots)
            ]
        )
        chain = Chain(new_pivots)
        chain = _redistribute_chain(chain, cummulative_weights)
        all_chains.append(chain)

    return all_chains


def _get_gradients_on_pivots(
    chain: Chain,
    loss_function: Callable[[ParameterVector], float],
    full_gradient_function: Callable[[ParameterVector], np.ndarray],
    calibrate_tangential: bool = False,
) -> np.ndarray:
    """Calculates gradient for every pivot on the chain w.r.t. the loss function
        using the gradient function.

    Args:
        chain: Chain to calculate the gradients on.
        loss_function: Loss function for which to calculate the gradient.
        full_gradient_function: Function to calculate the gradient w.r.t.
            the loss function for all parameters.
        calibrate_tangential: Flag to indicate whether next neighbor for finding
            tangential direction is calibrated with an additional loss evaluation.
            Defaults to False.
    """

    # We initialize with zeros, as we always want first and last gradient
    # to be equal to 0.
    gradients_on_pivots = np.zeros(shape=(chain.n_pivots, chain.n_params))

    for ii in range(1, chain.n_pivots - 1):
        before = chain.pivots[ii - 1]
        this = chain.pivots[ii]
        after = chain.pivots[ii + 1]
        #
        full_grad = full_gradient_function(this)
        #
        tan = this - before
        if calibrate_tangential and loss_function(after) > loss_function(before):
            tan = after - this
        tan /= np.linalg.norm(tan)
        tangential_grad = np.dot(full_grad, tan) * tan
        # save update
        gradients_on_pivots[ii] = full_grad - tangential_grad

    return gradients_on_pivots


def _redistribute_chain(
    chain: Chain,
    cummulative_weights: Optional[Weights] = None,
) -> Chain:
    """Helper function to re-distribute pivots along a Chain according
    to their weights on the chain."""
    path = ChainPath(chain)
    if cummulative_weights is not None:
        return path._get_chain_from_weights(cummulative_weights)
    else:
        return path.generate_uniform_chain(chain.n_pivots)

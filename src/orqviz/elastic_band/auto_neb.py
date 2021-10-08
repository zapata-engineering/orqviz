from typing import Callable, List, Optional

import numpy as np
from scipy.interpolate import interp1d

from ..aliases import ParameterVector
from .data_structures import Chain
from .neb import run_NEB


# Nudged-Elastic-Band
def run_AutoNEB(
    init_chain: Chain,
    loss_function: Callable[[ParameterVector], float],
    full_gradient_function: Optional[Callable[[ParameterVector], np.ndarray]] = None,
    n_cycles: int = 4,
    n_iters_per_cycle: int = 10,
    max_new_pivots: int = 1,
    weighted_redistribution: bool = False,
    insert_at_beginning: bool = False,
    percentage_tol: float = 0.2,
    absolute_tol: float = 0.0,
    eps: float = 0.1,
    learning_rate: float = 0.1,
    stochastic: bool = False,
    calibrate_tangential: bool = False,
    verbose: bool = False,
) -> List[Chain]:
    """Automatic wrapping of the NEB algorithm that dynamically inserts pivots
    to the piece-wise linear path being optimized in the loss function landscape.
    Returns a list of all chains during training where index -1
    is the final optimized chain.
    Reference paper:
        Essentially No Barriers in Neural Network Energy Landscape, arXiv:1803.00885.
    NOTE:
        There is a discrepancy between name used in the original paper (elastic band)
        and the one we use to define our data structure (chain), but we decided
        to use the later as we feel it's more appropriate for this data structure.

    Args:
        init_chain: Initial chain that is optimized with the algorithm.
        loss_function: Loss function that is used to optimize the chain.
        full_gradient_function: Function to calculate the gradient w.r.t.
            the loss function for all parameters. Defaults to None.
        n_cycles: Number of cycles between which new pivots can be inserted.
            Defaults to 4.
        n_iters_per_cycle: Number of optimization iterations per cycle.
            Defaults to 10.
        max_new_pivots: Maximum number of pivots inserted per cycle. Defaults to 1.
        weighted_redistribution: Flag to indicate whether pivots are uniformly
            re-distributed along the chain or according to their insertion position
            in the chain. Defaults to False.
        insert_at_beginning: Flag to indicate whether to insert pivots
            before first cycle. Defaults to False.
        percentage_tol: Percentage error threshold to insert new pivots.
            Called 'alpha' in the original paper. Be mindful of the magnitude and
            sign of typical loss values. Additive to absolute error. Defaults to 0.2.
        absolute_tol: Absolute error threshold to insert new pivots.
            Additive to percentage error. Defaults to 0.0.
        eps: Stencil for finite difference gradient if gradient_function
            is not provided. Defaults to 0.1.
        learning_rate: Learning rate/ step size for the gradient
            descent optimization. Defaults to 0.1.
        stochastic: Flag to indicate whether to perform stochastic gradient descent
            if gradient_function is not provided. It is less stable but much faster.
            Defaults to False.
        calibrate_tangential: Flag to indicate whether next neighbor for finding
            tangential direction is calibrated with an additional loss evaluation.
            Defaults to False.
        verbose: Flag for printing progress. Defaults to False.

    Returns:
        All chains during training
    """
    chain = init_chain

    all_chains = [chain]
    if insert_at_beginning:
        # find new pivots
        chain = _insert_pivots_to_improve_approximation(
            chain=chain,
            loss_function=loss_function,
            max_new_pivots=max_new_pivots,
            percentage_tol=percentage_tol,
            absolute_tol=absolute_tol,
        )

    for cycle in range(n_cycles):
        cummulative_weights = chain.get_weights() if weighted_redistribution else None
        optimization_history = run_NEB(
            init_chain=chain,
            loss_function=loss_function,
            full_gradient_function=full_gradient_function,
            n_iters=n_iters_per_cycle,
            cummulative_weights=cummulative_weights,
            eps=eps,
            learning_rate=learning_rate,
            stochastic=stochastic,
            calibrate_tangential=calibrate_tangential,
            verbose=verbose,
        )
        chain = optimization_history[-1]
        all_chains += optimization_history

        if cycle != n_cycles - 1:
            # find new pivots
            chain = _insert_pivots_to_improve_approximation(
                chain=chain,
                loss_function=loss_function,
                max_new_pivots=max_new_pivots,
                percentage_tol=percentage_tol,
                absolute_tol=absolute_tol,
            )
            all_chains.append(chain)

    return all_chains


def _insert_pivots_to_improve_approximation(
    chain: Chain,
    loss_function: Callable[[ParameterVector], float],
    max_new_pivots: int = 1,
    percentage_tol: float = 0.2,
    absolute_tol: float = 0.0,
) -> Chain:
    """Method to check where the piece-wise linear approximation of the current Chain
        with respect to the loss function is not good enough. If that's the case,
        it adds new pivots to the chain.

    Args:
        chain: Current Chain
        loss_function: Loss function for the NEB training
        max_new_pivots: Maximum number of pivots inserted to Chain. Defaults to 1.
        percentage_tol: Percentage error threshold to insert new pivots.
            Be mindful of the magnitude and sign of typical loss values.
            Additive to absolute error. Defaults to 0.2.
        absolute_tol: Absolute error threshold to insert new pivots.
            Additive to percentage error. Defaults to 0.0.
    """

    chain_weights = chain.get_weights()

    losses = chain.evaluate_on_pivots(loss_function)
    linear_weight_loss_map = interp1d(chain_weights, losses, kind="linear")
    linear_weight_chain_map = interp1d(
        chain_weights, chain.pivots, kind="linear", axis=0
    )
    # Check between each pair of pivots if an additional pivot
    # would improve the approximation
    trial_cum_weights = [
        (chain_weights[ii + 1] + chain_weights[ii]) / 2
        for ii in range(len(chain_weights) - 1)
    ]

    trial_points = linear_weight_chain_map(trial_cum_weights)
    trial_losses = np.array([loss_function(point) for point in trial_points])
    interpolated_losses = linear_weight_loss_map(trial_cum_weights)

    residuals = trial_losses - interpolated_losses
    significant_residuals = np.where(
        np.abs(residuals) > percentage_tol * interpolated_losses + absolute_tol,
        residuals,
        0,
    )
    new_pivots = chain.pivots

    if not np.isclose(significant_residuals, 0).all():
        largest_residual_positions = np.argsort(significant_residuals)[::-1]
        # sorts the most important indices backwards to insert correctly into the chain
        indices_to_select = sorted(
            largest_residual_positions[:max_new_pivots], reverse=True
        )
        for piv in indices_to_select:
            if np.isclose(trial_losses[piv], 0):
                continue
            new_cum_weight = trial_cum_weights[piv]
            insert_index = next(
                k for k, w in enumerate(chain_weights) if w > new_cum_weight
            )
            new_pivots = np.insert(
                new_pivots,
                insert_index,
                linear_weight_chain_map(new_cum_weight),
                axis=0,
            )

    return Chain(new_pivots)

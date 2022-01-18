import os

import matplotlib.pyplot as plt
import numpy as np

from orqviz.elastic_band import Chain, plot_all_chains_losses, run_AutoNEB, run_NEB
from orqviz.io import load_viz_object, save_viz_object


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def test_NEB():
    pivots = np.array([[t, 1 - t] for t in np.linspace(0, 1, 10)])
    init_chain = Chain(pivots=pivots)

    n_iters = 10
    all_chains = run_NEB(
        init_chain,
        loss_function=SUM_OF_SINES,
        full_gradient_function=None,
        n_iters=n_iters,
    )

    assert len(all_chains) == n_iters + 1
    for chain in all_chains:
        np.testing.assert_array_almost_equal(chain.pivots[0], np.array([0, 1]))
        np.testing.assert_array_almost_equal(chain.pivots[-1], np.array([1, 0]))


def test_AutoNEB():
    pivots = np.array([[t, 1 - t] for t in np.linspace(0, 1, 10)])
    init_chain = Chain(pivots=pivots)

    n_cycles = 4
    max_new_pivots = 2
    all_chains = run_AutoNEB(
        init_chain,
        loss_function=SUM_OF_SINES,
        full_gradient_function=None,
        n_cycles=n_cycles,
        n_iters_per_cycle=10,
        max_new_pivots=max_new_pivots,
    )

    for chain in all_chains:
        np.testing.assert_array_almost_equal(chain.pivots[0], np.array([0, 1]))
        np.testing.assert_array_almost_equal(chain.pivots[-1], np.array([1, 0]))

    save_viz_object(all_chains[-1], "test")
    loaded_chain = load_viz_object("test")
    os.remove("test")
    np.testing.assert_array_almost_equal(loaded_chain.pivots[0], np.array([0, 1]))
    np.testing.assert_array_almost_equal(loaded_chain.pivots[-1], np.array([1, 0]))
    np.testing.assert_array_almost_equal(loaded_chain.pivots, all_chains[-1].pivots)
    assert (
        all_chains[0].n_pivots
        <= all_chains[-1].n_pivots
        <= all_chains[0].n_pivots + (n_cycles - 1) * max_new_pivots
    )

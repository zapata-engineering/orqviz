import matplotlib.pyplot as plt
import numpy as np

from orqviz.elastic_band import Chain, plot_all_chains_losses, run_AutoNEB


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def test_plot_all_chains_losses():
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

    plot_all_chains_losses(all_chains, SUM_OF_SINES)
    fig, ax = plt.subplots()
    plot_all_chains_losses(all_chains, SUM_OF_SINES, ax=ax)

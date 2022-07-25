import matplotlib.pyplot as plt
import numpy as np
import pytest

from orqviz.pca import (
    get_pca,
    perform_2D_pca_scan,
    plot_line_through_points_on_pca,
    plot_optimization_trajectory_on_pca,
    plot_pca_landscape,
    plot_scatter_points_on_pca,
)
from orqviz.scans.data_structures import Scan2DResult


def COST_FUNCTION(params):
    return np.sum(np.sin(params)) + np.sum(params**2) + 3 * params[1] - 10 * params[2]


@pytest.fixture
def all_points():
    return np.random.rand(98, 9)


@pytest.fixture
def pca(all_points):

    return get_pca(all_points, components_ids=(0, 1))


@pytest.fixture
def scan_results(pca):
    n_steps_x = 100
    n_steps_y = 20

    return perform_2D_pca_scan(
        pca_object=pca,
        loss_function=COST_FUNCTION,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
    )


def test_plot_pca_landscape(scan_results, pca):
    plot_pca_landscape(scan_results, pca_object=pca)
    fig, ax = plt.subplots(1, 1)
    plot_pca_landscape(scan_results, pca_object=pca, fig=fig, ax=ax)


def test_plot_optimization_trajectory_on_pca(all_points, pca):
    plot_optimization_trajectory_on_pca(all_points, pca_object=pca)
    fig, ax = plt.subplots(1, 1)
    plot_optimization_trajectory_on_pca(all_points, pca_object=pca, ax=ax)


def test_plot_scatter_points_on_pca(all_points, pca):
    plot_scatter_points_on_pca(all_points, pca_object=pca)
    fig, ax = plt.subplots(1, 1)
    plot_scatter_points_on_pca(all_points, pca_object=pca, ax=ax)


def test_plot_line_through_points_on_pca(all_points, pca):
    plot_line_through_points_on_pca(all_points, pca_object=pca)
    fig, ax = plt.subplots(1, 1)
    plot_line_through_points_on_pca(all_points, pca_object=pca, ax=ax)

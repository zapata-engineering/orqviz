import pytest
import numpy as np
from orqviz.scans.data_structures import Scan2DResult
from orqviz.pca import (
    perform_2D_pca_scan,
    get_pca,
    PCAobject,
    plot_pca_landscape,
    plot_scatter_points_on_pca,
    plot_optimization_trajectory_on_pca,
    plot_line_through_points_on_pca,
)
import matplotlib.pyplot as plt


def COST_FUNCTION(params):
    return np.sum(np.sin(params)) + np.sum(params ** 2) + 3 * params[1] - 10 * params[2]


def test_pca():
    all_points = np.random.rand(98, 9)
    for components_ids in [(0, 1), (1, 5)]:
        pca = get_pca(all_points, components_ids=components_ids)

        n_steps_x = 100
        n_steps_y = 20

        scan_results = perform_2D_pca_scan(
            pca_object=pca,
            loss_function=COST_FUNCTION,
            n_steps_x=n_steps_x,
            n_steps_y=n_steps_y,
        )
        assert scan_results.params_grid.shape == (n_steps_y, n_steps_x, 2)
        np.testing.assert_array_equal(scan_results.direction_x, np.array([1, 0]))
        np.testing.assert_array_equal(scan_results.direction_y, np.array([0, 1]))
        assert pca.get_transformed_points().shape == (
            len(all_points),
            max(pca.components_ids) + 1,
        )
        assert pca.get_transformed_points(all_points[:24]).shape == (
            24,
            max(pca.components_ids) + 1,
        )

        plot_pca_landscape(scan_results, pca_object=pca)
        fig, ax = plt.subplots(1, 1)
        plot_pca_landscape(scan_results, pca_object=pca, fig=fig, ax=ax)

        for func in [
            plot_optimization_trajectory_on_pca,
            plot_scatter_points_on_pca,
            plot_line_through_points_on_pca,
        ]:
            func(all_points, pca_object=pca, ax=ax)

    new_components_ids = (2, 3)
    pca.set_component_ids(new_components_ids)

    scan_results = perform_2D_pca_scan(
        pca_object=pca,
        loss_function=COST_FUNCTION,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
    )
    assert scan_results.params_grid.shape == (n_steps_y, n_steps_x, 2)
    np.testing.assert_array_equal(scan_results.direction_x, np.array([1, 0]))
    np.testing.assert_array_equal(scan_results.direction_y, np.array([0, 1]))
    assert pca.get_transformed_points().shape == (
        len(all_points),
        max(pca.components_ids) + 1,
    )
    assert pca.get_transformed_points(all_points[:24]).shape == (
        24,
        max(pca.components_ids) + 1,
    )

    plot_pca_landscape(scan_results, pca_object=pca)
    fig, ax = plt.subplots(1, 1)
    plot_pca_landscape(scan_results, pca_object=pca, fig=fig, ax=ax)

    for func in [
        plot_optimization_trajectory_on_pca,
        plot_scatter_points_on_pca,
        plot_line_through_points_on_pca,
    ]:
        func(all_points, pca_object=pca, ax=ax)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orqviz.pca import get_pca, perform_2D_pca_scan
from orqviz.scans.data_structures import Scan2DResult


def COST_FUNCTION(params):
    return np.sum(np.sin(params)) + np.sum(params**2)


@pytest.mark.parametrize(
    "all_points,components_ids",
    [[np.random.rand(98, 9), (0, 1)], [np.random.rand(27, 10, 4), (1, 5)]],
)
def test_pca(all_points, components_ids):
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

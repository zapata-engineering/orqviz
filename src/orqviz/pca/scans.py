from typing import Callable, List, Tuple, Union

import numpy as np

from ..aliases import ParameterVector
from ..scans import Scan2DResult, perform_2D_scan
from .data_structures import PCAobject


def perform_2D_pca_scan(
    pca_object: PCAobject,
    loss_function: Callable[[ParameterVector], float],
    n_steps_x: int = 20,
    n_steps_y: int = None,
    offset: Union[Tuple[float, float], float] = (-1.0, 1.0),
    verbose: bool = False,
) -> Scan2DResult:
    """Function to perform a 2D scan on a loss function landscape
        according in directions of PCA components.

    Args:
        all_points: Points on which PCA was fitted and around which the scan
            is performed.
        loss_function: Loss function which is scanned.
        pca: PCA object that was fitted on all_points.
            Its components are used to decide scan directions.
        components_ids: Which components of the PCA object are used as scan directions.
            Defaults to (0, 1).
        n_steps_x: Number of grid points in x-direction to perform the scan.
            Defaults to 20.
        n_steps_y: Number of grid points in y-direction to perform the scan.
            If set to None, it is set to n_steps_x. Defaults to None.
        offset: Offset in x-y directions added to the scan range on top of
            the range that is necessary to display all_points. Defaults to (-1.0, 1.0).
        verbose: Flag for printing progress. Defaults to False.
    """

    if not isinstance(offset, tuple):
        offset = (-np.abs(offset), np.abs(offset))

    if n_steps_y is None:
        n_steps_y = n_steps_x

    end_points_x, end_points_y = pca_object._get_endpoints_from_pca(offset)

    def pca_loss_function(xy_params):
        pca_params = np.zeros(pca_object.pca.n_components)
        pca_params[pca_object.components_ids[0]] = xy_params[0]
        pca_params[pca_object.components_ids[1]] = xy_params[1]
        probe_parameters = pca_object.pca.inverse_transform(pca_params)
        return loss_function(probe_parameters)

    return perform_2D_scan(
        origin=np.array([0, 0]),
        loss_function=pca_loss_function,
        direction_x=np.array([1, 0]),
        direction_y=np.array([0, 1]),
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
        verbose=verbose,
    )

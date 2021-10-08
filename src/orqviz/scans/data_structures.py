from typing import NamedTuple, Optional

import numpy as np

from ..aliases import ArrayOfParameterVectors, GridOfParameterVectors, ParameterVector
from ..geometric import get_coordinates_on_direction


class Scan1DResult(NamedTuple):
    """Datatype for 1D scans to combine the scan result and scan instruction."""

    params_list: ArrayOfParameterVectors
    direction: np.ndarray
    values: np.ndarray
    origin: Optional[ParameterVector] = None

    def _get_coordinates_on_direction(self, in_units_of_direction: bool = False):
        """Projects parameters in params_vector and projects them on the direction_x."""
        return get_coordinates_on_direction(
            self.params_list,
            self.direction,
            origin=self.origin,
            in_units_of_direction=in_units_of_direction,
        )


class Scan2DResult(NamedTuple):
    """Datatype for 2D scans to combine the scan result and scan instruction."""

    params_grid: GridOfParameterVectors
    direction_x: np.ndarray
    direction_y: np.ndarray
    values: np.ndarray
    origin: Optional[ParameterVector] = None

    def _get_coordinates_on_directions(self, in_units_of_direction: bool = False):
        """Projects parameters in directions x and y."""
        x_values = get_coordinates_on_direction(
            self.params_grid[0, :],
            self.direction_x,
            origin=self.origin,
            in_units_of_direction=in_units_of_direction,
        )
        y_values = get_coordinates_on_direction(
            self.params_grid[:, 0],
            self.direction_y,
            origin=self.origin,
            in_units_of_direction=in_units_of_direction,
        )
        return x_values, y_values


def clone_Scan1DResult_with_different_values(
    base_scan: Scan1DResult, values: np.ndarray
) -> Scan1DResult:
    """Helper function that returns a copy of a Scan1DResult object with new values"""
    class_dict = base_scan._asdict()
    class_dict["values"] = values
    return Scan1DResult(**class_dict)


def clone_Scan2DResult_with_different_values(
    base_scan: Scan2DResult, values: np.ndarray
) -> Scan2DResult:
    """Helper function that returns a copy of a Scan2DResult object with new values"""
    class_dict = base_scan._asdict()
    class_dict["values"] = values
    return Scan2DResult(**class_dict)

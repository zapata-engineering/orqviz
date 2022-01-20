import os

import numpy as np
import pytest

from orqviz.elastic_band.data_structures import Chain
from orqviz.hessians import get_Hessian
from orqviz.io import OrqVizObject, load_viz_object, save_viz_object
from orqviz.scans import perform_1D_scan, perform_2D_scan


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def test_saving_and_loading_datatypes():
    origin = np.random.rand(2)
    direction_x = np.random.rand(2)
    direction_y = np.random.rand(2)
    n_steps_x = 2

    scan1d = perform_1D_scan(
        loss_function=SUM_OF_SINES,
        origin=origin,
        direction=direction_x,
        n_steps=n_steps_x,
    )

    scan2d = perform_2D_scan(
        origin=origin,
        loss_function=SUM_OF_SINES,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
    )

    hessian = get_Hessian(params=origin, loss_function=SUM_OF_SINES)

    chain = Chain(np.linspace(origin, origin + direction_x, num=5))

    for data_object in [scan1d, scan2d, hessian, chain]:
        save_viz_object(data_object, "test")
        loaded_data_object = load_viz_object("test")
        os.remove("test")
        assert isinstance(loaded_data_object, OrqVizObject.__args__)
        assert type(loaded_data_object) == type(data_object)

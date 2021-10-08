import pickle
from typing import Union

from .elastic_band.data_structures import Chain
from .hessians.data_structures import HessianEigenobject
from .scans.data_structures import Scan1DResult, Scan2DResult

OrqVizObject = Union[Scan1DResult, Scan2DResult, HessianEigenobject, Chain]


def save_viz_object(viz_object: OrqVizObject, filename: str):
    """Save datatype to a with pickle"""

    with open(filename, "wb") as f:
        pickle.dump(viz_object, f)


def load_viz_object(filename: str) -> OrqVizObject:
    """Load datatype from a file with pickle"""

    with open(filename, "rb") as f:
        loaded_object = pickle.load(f)

    return loaded_object

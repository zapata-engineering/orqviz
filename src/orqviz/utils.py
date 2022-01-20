import warnings
from typing import Union

from .elastic_band.data_structures import Chain
from .hessians.data_structures import HessianEigenobject
from .io import load_viz_object as _load_viz_object
from .io import save_viz_object as _save_viz_object
from .scans.data_structures import Scan1DResult, Scan2DResult

OrqVizObject = Union[Scan1DResult, Scan2DResult, HessianEigenobject, Chain]


def save_viz_object(viz_object: OrqVizObject, filename: str) -> None:
    """Save datatype to a with pickle"""
    warnings.warn(
        """orqviz.utils.save_viz_object is deprecated,
        please use orqviz.io.save_viz_object""",
        DeprecationWarning,
    )
    _save_viz_object(viz_object, filename)


def load_viz_object(filename: str) -> OrqVizObject:
    """Load datatype from a file with pickle"""
    warnings.warn(
        """orqviz.utils.load_viz_object is deprecated,
        please use orqviz.io.load_viz_object""",
        DeprecationWarning,
    )
    return _load_viz_object(filename)

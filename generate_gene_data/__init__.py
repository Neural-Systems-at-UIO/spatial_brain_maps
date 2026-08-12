__version__ = "0.1.0"

from .volume import id_to_volume, gene_to_volume, interpolate, write_nifti
from . import utilities


def id_to_points(*args, **kwargs):
    from .points import id_to_points as implementation

    return implementation(*args, **kwargs)


def gene_to_points(*args, **kwargs):
    from .points import gene_to_points as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "id_to_points",
    "gene_to_points",
    "id_to_volume",
    "gene_to_volume",
    "interpolate",
    "write_nifti",
    "utilities",
]

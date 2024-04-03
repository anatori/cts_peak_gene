from . import simu, method, data_loader, util

# expose functions so that scdrs.score_cell, scdrs.preprocess can be called
# from .method import score_cell
# from .pp import preprocess
from .version import __version__,__version_info__

__all__ = ["simu", "method", "data_loader", "util"]
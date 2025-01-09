"""
FRG is a Python package to deal with the functional renormalisation group for signal detection. It leverages the action of the RG on different sources of signal to compute a limit of detection and assess the presence of a signal.

The work is based on `recent theoretical advancements <https://arxiv.org/abs/2201.04250>`__ and `numerical works <https://arxiv.org/abs/2310.07499>`__.
"""

# Import the modules
from frg.distributions.distributions import MarchenkoPastur
from frg.utils.utils import get_cfg_defaults, get_logger

# Set the version number
__version__ = "0.9.0b"

# Set the author
__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"

# Set the license description
__license__ = "CEA Proprietary License"
__url__ = ""

# Package imports
__all__ = [
    "MarchenkoPastur",
    "get_cfg_defaults",
    "get_logger",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
]

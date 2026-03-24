"""
Utilities
---------

Utility functions and helpers to handle configuration files and logging.

Authors
-------

- Riccardo Finotello <riccardo.finotello@cea.fr>
- Parham Radpay <parhamradpay@gmail.com>

Maintainers
-----------

- Riccardo Finotello
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from yacs.config import CfgNode as CN

from frg import EmpiricalDistribution

if TYPE_CHECKING:
    from typing import Literal, TextIO

    from jaxtyping import Float

    # Dummy variables for jaxtyping to prevent Ruff F821 errors.
    # Defined strictly within TYPE_CHECKING so they cannot exist at runtime,
    # ensuring they never overwrite or conflict with actual code variables.
    p: int = 500
    h: int = 128
    w: int = 128
    c: int = 3


def get_cfg_defaults() -> CN:
    """
    Get the default configuration.

    Returns
    -------
    CfgNode
        The default configuration (YACS CfgNode)
    """

    cfg: CN = CN()

    # Distribution parameters
    cfg.DIST: CN = CN()
    cfg.DIST.NUM_SAMPLES: int = 1000
    cfg.DIST.VAR: float = 1.0
    cfg.DIST.RATIO: float = 0.5
    cfg.DIST.SEED: int = 42
    cfg.DIST.IS_POIS: bool = False
    cfg.DIST.POIS_DATA: bool = False
    cfg.DIST.POIS_LAM: float = 10.0
    cfg.DIST.POIS_MODE: Literal["centered", "non-centered", "mirrored"] = (
        "centered"
    )

    # Signal parameters
    cfg.SIG: CN = CN()
    cfg.SIG.INPUT: str | None = None
    cfg.SIG.SNR: float = 0.0

    # Potential parameters
    cfg.POT: CN = CN()
    cfg.POT.UV_SCALE: float = 1.0e-5
    cfg.POT.KAPPA_INIT: float = 1.0e-5
    cfg.POT.U2_INIT: float = 1.0e-5
    cfg.POT.U4_INIT: float = 1.0e-5
    cfg.POT.U6_INIT: float = 1.0e-5

    # Data parameters
    cfg.DATA: CN = CN()
    cfg.DATA.OUTPUT_DIR: str = "results"

    return cfg.clone()


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get the logger.

    Parameters
    ----------
    name : str
        The name of the logger (logging session)
    level : int
        The logging level:

            - logging.DEBUG = 10
            - logging.INFO = 20
            - logging.WARNING = 30
            - logging.ERROR = 40
            - logging.CRITICAL = 50

    Returns
    -------
    logging.Logger
        The logger
    """
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set up the format
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    formatter: logging.Formatter = logging.Formatter(
        fmt="{asctime} | [{levelname:^8s}] : {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    # Set the format
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def load_data(cfg: CN) -> EmpiricalDistribution:
    """
    Load the data from file.

    Parameters
    ----------
    cfg : CN
        The configuration file.

    .. warning:

        The image must be a B/W image (single channel) and present pixels in the range :math:`[0, 255]`.

    Returns
    -------
    EmpiricalDistribution
        The distribution
    """
    data: Path = Path(os.path.expandvars(cfg.SIG.INPUT)).absolute()
    if not data.exists():
        raise FileNotFoundError("Input data file %s does not exist!" % data)

    # Create the distribution
    if data.suffix.lower() == ".npy":  # covariance matrix
        data: Float[np.ndarray, "p, p"] = np.load(data)
        if data.ndim != 2:
            raise ValueError(
                "Covariance matrix must be 2-dimensional but %d-dimensional found!"
                % data.ndim
            )
        if data.shape[0] != data.shape[1]:
            raise ValueError(
                "Covariance matrix must be square but shape %s found!"
                % (data.shape,)
            )
        dist: EmpiricalDistribution = EmpiricalDistribution.from_covariance(
            cov=data, cfg=cfg
        ).fit()
    elif data.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        import imageio.v3 as iio  # import images

        img: Float[np.ndarray, "h, w, c"] | Float[np.ndarray, "h, w"] = (
            iio.imread(data)
        )
        if img.ndim > 2:
            img = img.mean(axis=-1)
        img = img.astype(float)
        img -= img.mean()  # centre the image
        img /= img.std()  # scale the image
        dist: EmpiricalDistribution = EmpiricalDistribution.from_config(
            cfg
        ).fit(X=img, snr=cfg.SIG.SNR, fac=0.3)

    return dist
    return dist

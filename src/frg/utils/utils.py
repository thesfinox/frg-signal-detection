"""
Utility functions and helpers to handle configuration files and logging.
"""

import logging
import os
from pathlib import Path

import numpy as np
from yacs.config import CfgNode as CN

from frg.distributions.distributions import EmpiricalDistribution


def get_cfg_defaults() -> CN:
    """
    Get the default configuration.

    Returns
    -------
    CN
        The default configuration (YACS CfgNode)
    """

    cfg = CN()

    # Distribution parameters
    cfg.DIST = CN()
    cfg.DIST.NUM_SAMPLES = 1000
    cfg.DIST.SIGMA = 1.0
    cfg.DIST.RATIO = 0.5
    cfg.DIST.SEED = 42

    # Signal parameters
    cfg.SIG = CN()
    cfg.SIG.INPUT = None
    cfg.SIG.SNR = 0.0

    # Potential parameters
    cfg.POT = CN()
    cfg.POT.UV_SCALE = 1.0e-5
    cfg.POT.KAPPA_INIT = 1.0e-5
    cfg.POT.U2_INIT = 1.0e-5
    cfg.POT.U4_INIT = 1.0e-5
    cfg.POT.U6_INIT = 1.0e-5

    # Data parameters
    cfg.DATA = CN()
    cfg.DATA.OUTPUT_DIR = "results"

    # Plots parameters
    cfg.PLOTS = CN()
    cfg.PLOTS.OUTPUT_DIR = "plots"

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
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Set up the format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
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

    Returns
    -------
    EmpiricalDistribution
        The distribution
    """
    data = Path(os.path.expandvars(cfg.SIG.INPUT))
    if not data.exists():
        raise FileNotFoundError("Input data file %s does not exist!" % data)

        # Create the distribution
    if "npy" in data.suffix.lower():
        # Load covariance matrix
        data = np.load(data)
        dist = EmpiricalDistribution.from_covariance(data).fit()
    elif ("png" in data.suffix.lower()) or ("jpg" in data.suffix.lower()):
        # Load image
        from PIL import Image

        img = np.array(Image.open(data)) / 255.0
        img -= img.mean()  # centre the image
        img /= img.std()  # scale the image
        dist = EmpiricalDistribution.from_config(cfg).fit(
            X=img,
            snr=cfg.SIG.SNR,
        )

    return dist

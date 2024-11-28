#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Compute the flow of the FRG for a small deviation from a Marchenko-Pastur distribution.
"""

import argparse
import logging
import sys
from pathlib import Path
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from yacs.config import CfgNode as CN

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Compute the flow of the FRG for a small deviation from a Marchenko-Pastur distribution."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def get_cfg_defaults() -> CN:
    """
    Get the default configuration.

    Returns
    -------
    CN
        The default configuration (YACS CfgNode)
    """

    cfg = CN()

    # Covariance matrix
    cfg.COV = CN()
    cfg.COV.PATH = None

    # Distribution parameters
    cfg.DIST = CN()
    cfg.DIST.NUM_SAMPLES = 1000
    cfg.DIST.SIGMA = 1.0
    cfg.DIST.RATIO = 0.5
    cfg.DIST.SEED = 42

    # Signal parameters
    cfg.SIG = CN()
    cfg.SIG.IMAGE = None
    cfg.SIG.SNR = 0.0

    return cfg.clone()


def get_logger(name: str) -> logging.Logger:
    """
    Get the logger.

    Parameters
    ----------
    name : str
        The name of the logger (logging session)

    Returns
    -------
    logging.Logger
        The logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Set up the format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | [%(levelname)8s] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class MarchenkoPastur:
    """Marchenko-Pastur distribution"""

    def __init__(self, ratio: float, sigma: float):
        """
        Parameters
        ----------
        ratio : float
            The ratio between the number of variables and the dimension of the sample/population.

            .. note::

                Let :math:`X \\in \\mathbb{R}^{n \\times p}` be a real matrix where samples are **row** vectors, then the ``ratio`` parameter is :math:`\\frac{p}{n}`.

        sigma : float
            The standard deviation of the distribution.

        Raises
        ------
        ValueError
            If ratio is not strictly positive
        ValueError
            If the lowest eigenvalue is higher than the highest one
        UserWarning
            If ratio is higher than 1 because of numerical instabilities
        """
        # Sanitise the input
        if ratio <= 0.0:
            raise ValueError(
                "Ratio must be strictly positive but got %f <= 0" % ratio
            )
        if ratio >= 1.0:
            warn(
                "Ratios higher than 1 may result in numerical instabilities. We got %f >= 1"
                % ratio,
                category=UserWarning,
            )

        # Store the parameters
        self.ratio = ratio
        self.sigma = sigma

        # Compute the highest and lowest eigenvalues
        self.lplus = self.sigma**2 * (1.0 + np.sqrt(self.ratio)) ** 2
        self.lminus = self.sigma**2 * (1.0 - np.sqrt(self.ratio)) ** 2
        self.m2 = 1 / self.lplus  # Mass
        if self.lminus > self.lplus:
            raise ValueError(
                "The lowest eigenvalue must be lower than the highest one, but got lminus = %f > %f = lplus!"
                % (self.lminus, self.lplus)
            )

    def pdf(
        self, x: float | ArrayLike, shift: float = 0.0
    ) -> float | ArrayLike:
        """
        Compute the PDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the PDF.
        shift : float, optional
            The shift of the distribution in the x axis, by default 0.

        Returns
        -------
        float | ArrayLike
            The value(s) of the PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.pdf, otypes=[np.float64])(x, shift)

        if (x <= self.lminus - shift) or (x >= self.lplus - shift):
            return 0.0
        num = np.sqrt(
            max(0, self.lplus - shift - x) * max(0, x + shift - self.lminus)
        )
        den = 2.0 * np.pi * self.sigma**2 * self.ratio * (x + shift)
        return num / den

    def cdf(
        self, x: float | ArrayLike, shift: float = 0.0
    ) -> float | ArrayLike:
        """
        Compute the CDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the CDF.
        shift : float, optional
            The shift of the distribution in the x axis, by default 0.

        Returns
        -------
        float | ArrayLike
            The value(s) of the CDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.cdf, otypes=[np.float64])(x, shift)

        if x <= self.lminus - shift:
            return 0.0
        if x >= self.lplus - shift:
            return 1.0
        return quad(lambda y: self.pdf(y, shift), self.lminus - shift, x)[0]

    def ipdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """
        Compute the PDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float | ArrayLike
            The value(s) of the PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.ipdf, otypes=[np.float64])(x)

        return (
            self.pdf(1 / (x + self.m2), shift=self.lminus) / (x + self.m2) ** 2
        )

    def canonical_dimensions(self) -> dict[str, ArrayLike]:
        """
        Compute the canonical dimensions of the distribution.

        Returns
        -------
        Dict[str, ArrayLike]
            The canonical dimensions for the couplings:

            - 'u2': the mass term
            - 'u4': the quartic interaction
            - 'u6': the sextic interaction
        """
        pass


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger = get_logger(__name__)
    logger.info("Starting...")

    # Open the configuration file
    cfg_file = Path(a.config)
    logger.debug("Configuration file: %s" % cfg_file)
    if not cfg_file.exists():
        logger.error("Configuration file %s not found!" % cfg_file)
        raise FileNotFoundError("Configuration file %s not found!" % cfg_file)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(a.args)
    cfg.freeze()

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    if a.analytic:
        logger.info("Running an analytic simulation...")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("config", help="Configuration file")
    parser.add_argument(
        "--print_config", action="store_true", help="Print configuration"
    )
    parser.add_argument(
        "--analytic", action="store_true", help="Run an analytic simulation"
    )
    parser.add_argument(
        "--args",
        nargs="+",
        default=[],
        help="Additional configuration arguments",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

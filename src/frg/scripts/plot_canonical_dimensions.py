#! /usr/bin/env python3
r"""Plot the canonical dimensions on the momentum or eigenvalue spectrum.

This script wraps two plotting functions:

- :func:`~frg.utils.analysis.plot_canonical_dimensions`: plots the canonical dimensions :math:`\text{dim}(u_2)`, :math:`\text{dim}(u_4)`, and :math:`\text{dim}(u_6)` as a function of the momentum scale :math:`k^2`.

- :func:`~frg.utils.analysis.plot_canonical_dimensions_eigenvalues`: plots the same quantities remapped to the eigenvalue axis :math:`\lambda` via the inverse change of variables

    .. math::

        \lambda = \frac{1}{k^2 + m^2} + \lambda_-

Both analytic (Marchenko-Pastur) and empirical distributions are supported.  For the eigenvalue mode, the distribution parameters ``ratio`` and ``var`` are automatically parsed from the result file name (``ratio=…``, ``var=…`` tokens) or can be supplied explicitly via ``--ratio`` and ``--var``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde

from frg import (
    MarchenkoPastur,
    get_logger,
    plot_canonical_dimensions,
    plot_canonical_dimensions_eigenvalues,
)
from frg.distributions.distributions import Distribution

mpl.rc("font", size=16)

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float

    n_steps: int = 1

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = (
    "Plot the canonical dimensions on the momentum or eigenvalue spectrum."
)
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


class _EmpiricalDistProxy(Distribution):
    r"""Minimal distribution proxy reconstructed from stored eigenvalues.

    This lightweight object provides the three attributes consumed by
    :func:`~frg.utils.analysis.plot_canonical_dimensions_eigenvalues`:

    - ``m2``: inverse of the spectral range (from the JSON result file).
    - ``lminus``: lower edge of the bulk, :math:`\sigma^2 (1 - \sqrt{q})^2`.
    - ``pdf``: KDE built from the stored bulk eigenvalues.

    Parameters
    ----------
    evl : ndarray of floats
        The bulk eigenvalues (``"evl"`` key from the JSON result file).
    m2 : float
        The mass stored in the JSON result file.
    ratio : float
        Marchenko-Pastur ratio :math:`q = p / n`.
    var : float
        Marchenko-Pastur variance :math:`\sigma^2`.
    """

    def __init__(
        self,
        evl: Float[np.ndarray, n_steps],
        m2: float,
        ratio: float,
        var: float,
    ) -> None:
        super().__init__()
        self.m2: float = m2
        self.lminus: float = float(var * (1.0 - np.sqrt(ratio)) ** 2)
        self.lplus: float = float(max(evl))
        self._kde = gaussian_kde(
            evl,
            bw_method=lambda obj: np.power(obj.n, -1.0 / (obj.d + 4.0)) * 0.3,
        )
        self._norm: float = float(
            self._kde.integrate_box_1d(self.lminus, self.lplus),
        )

    def pdf(self, x):  # noqa: D102
        if not isinstance(x, float):
            return np.vectorize(self.pdf, otypes=[np.float64])(x)
        if x < self.lminus or x > self.lplus:
            return 0.0
        return float(self._kde(x)[0]) / self._norm

    def cdf(self, x):  # noqa: D102
        raise NotImplementedError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__,
        epilog=__epilog__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="JSON file produced by frg-canonical-dimensions",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory for the plots",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix appended to the output file name",
    )
    parser.add_argument(
        "--analytic",
        action="store_true",
        help="Treat the simulation as analytic (disables interpolation / ROI annotations)",
    )
    parser.add_argument(
        "--eigenvalues",
        action="store_true",
        help=(
            "Plot on the eigenvalue axis λ instead of the momentum axis k². "
            "Pass --ratio and --var if they cannot be parsed from the file name."
        ),
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Marchenko-Pastur ratio p/n (used with --eigenvalues)",
    )
    parser.add_argument(
        "--var",
        type=float,
        default=None,
        help="Marchenko-Pastur variance σ² (used with --eigenvalues)",
    )
    parser.add_argument(
        "-v",
        dest="verb",
        action="count",
        default=0,
        help="Verbosity level",
    )
    return parser.parse_args(argv)


def _resolve_ratio_var(
    args: argparse.Namespace,
    stem: str,
    logger: Logger,
) -> tuple[float, float]:
    """Resolve ratio and var from CLI args or file-name patterns.

    Returns
    -------
    tuple[float, float]
        The resolved ratio and var values.

    Raises
    ------
    ValueError
        If ratio or var cannot be determined from args or file name.
    """
    ratio: float | None = args.ratio
    var: float | None = args.var

    if ratio is None:
        m = re.search(r"ratio=([0-9]+[.][0-9]*)", stem)
        if m:
            ratio = float(m.group(1))
    if var is None:
        m = re.search(r"var=([0-9]+[.][0-9]*)", stem)
        if m:
            var = float(m.group(1))

    if ratio is None or var is None:
        logger.error(
            "Could not determine ratio/var. Pass --ratio and --var explicitly.",
        )
        raise ValueError("Missing --ratio / --var for the eigenvalue plot.")

    logger.info("Using ratio=%s, var=%s", ratio, var)
    return float(ratio), float(var)


def _build_dist(
    args: argparse.Namespace,
    data: dict,
    stem: str,
    logger: Logger,
) -> Distribution:
    """Build the distribution object needed for the eigenvalue-spectrum plot.

    Returns
    -------
    Distribution
        Either a :class:`~frg.distributions.distributions.MarchenkoPastur`
        (analytic mode) or an :class:`_EmpiricalDistProxy` (empirical mode).

    Raises
    ------
    ValueError
        If ratio/var cannot be resolved, or if ``'evl'`` is missing from the
        result file in empirical mode.
    """
    ratio, var = _resolve_ratio_var(args, stem, logger)

    if args.analytic:
        logger.info("Reconstructing MarchenkoPastur distribution...")
        return MarchenkoPastur(ratio=ratio, var=var)

    evl_raw = data.get("evl")
    if evl_raw is None:
        logger.error(
            "The JSON key 'evl' is None; cannot reconstruct the empirical "
            "distribution for the eigenvalue plot.",
        )
        raise ValueError(
            "'evl' is None in the JSON file; cannot build the empirical distribution."
        )
    evl: Float[np.ndarray, n_steps] = np.array(evl_raw)
    logger.info(
        "Reconstructing empirical distribution from %d eigenvalues...",
        len(evl),
    )
    return _EmpiricalDistProxy(
        evl=evl, m2=float(data["m2"]), ratio=ratio, var=var
    )


def main(argv: list[str] | None = None) -> int | str:
    """Run the canonical dimensions plotting script.

    Parameters
    ----------
    argv : list[str], optional
        The command-line arguments.

    Returns
    -------
    int | str
        The exit code or an error message.

    Raises
    ------
    ValueError
        If the input file does not exist or required parameters are missing.
    """
    args: argparse.Namespace = _parse_args(argv)

    logger_level: int = 10 * (4 - args.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Validate and load input
    inp_file: Path = Path(os.path.expandvars(args.input)).absolute()
    if not inp_file.exists():
        logger.error("%s does not exist!", inp_file)
        raise ValueError(f"{inp_file} does not exist!")

    logger.info("Loading %s ...", inp_file)
    with inp_file.open() as f:
        data: dict[str, Float[np.ndarray, n_steps]] = json.load(f)
    for key in ("k2", "dimu2", "dimu4", "dimu6", "dist"):
        data[key] = np.array(data[key])

    if not args.eigenvalues:
        # Momentum-spectrum plot (default)
        logger.info("Plotting on the momentum spectrum (k²)...")
        plot_canonical_dimensions(
            data,
            suffix=args.suffix,
            analytic=args.analytic,
            output_dir=args.output,
        )
        _stem = (
            "canonical_dimensions"
            if args.suffix is None
            else f"canonical_dimensions_{args.suffix}"
        )
        logger.info("Plot saved to %s", Path(args.output) / f"{_stem}.pdf")
        return 0

    # Eigenvalue-spectrum plot
    logger.info("Plotting on the eigenvalue spectrum (λ)...")
    dist: Distribution = _build_dist(args, data, inp_file.stem, logger)
    plot_canonical_dimensions_eigenvalues(
        data,
        dist=dist,
        suffix=args.suffix,
        analytic=args.analytic,
        output_dir=args.output,
    )
    _stem = (
        "canonical_dimensions_eigenvalues"
        if args.suffix is None
        else f"canonical_dimensions_eigenvalues_{args.suffix}"
    )
    logger.info("Plot saved to %s", Path(args.output) / f"{_stem}.pdf")
    return 0


def cli():  # noqa: D103
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

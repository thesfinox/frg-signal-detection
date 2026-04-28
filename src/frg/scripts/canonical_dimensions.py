#! /usr/bin/env python3
"""Compute the canonical dimensions of the couplings in a theory with given momenta distribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from frg import (
    EmpiricalDistribution,
    MarchenkoPastur,
    get_cfg_defaults,
    get_logger,
    load_data,
)

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float
    from yacs.config import CfgNode

    n_dof: int = 1000

__author__: str = "Riccardo Finotello and Parham Radpay"
__email__: str = "riccardo.finotello@cea.fr; parham.radpay@gmail.com"
__description__: str = "Compute the canonical dimensions of the couplings in a theory with given momenta distribution."
__epilog__: str = (
    "For bug reports and info: " + __author__ + " <" + __email__ + ">"
)


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
    parser.add_argument("--config", required=False, help="Configuration file")
    parser.add_argument(
        "--analytic",
        action="store_true",
        help="Run an analytic simulation",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print configuration",
    )
    parser.add_argument(
        "--suffix",
        nargs="+",
        choices=["nsamples", "ratio", "seed", "var", "lam", "mode"],
        help="Type of suffix used in the output files. Must be a list containing one or more of the following: nsamples, ratio, seed, var, lam, mode",
    )
    parser.add_argument(
        "--args",
        nargs="+",
        default=[],
        help="Additional configuration arguments (see YACS documentation)",
    )
    parser.add_argument(
        "-v",
        dest="verb",
        action="count",
        default=0,
        help="Verbosity level",
    )
    return parser.parse_args(argv)


def _load_config(a: argparse.Namespace, logger: Logger) -> CfgNode:
    """Load the configuration file.

    Returns
    -------
    CfgNode
        The configuration node.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    cfg: CfgNode = get_cfg_defaults()
    if a.config is not None:
        logger.debug("Configuration file: %s" % a.config)
        cfg_file = Path(os.path.expandvars(a.config)).absolute()
        if cfg_file.exists():
            logger.debug("Configuration file exists!")
            cfg.merge_from_file(cfg_file)
        else:
            logger.error("Configuration file %s does not exist!", cfg_file)
            raise FileNotFoundError(
                "Configuration file %s does not exist!" % cfg_file,
            )
    cfg.merge_from_list(a.args)
    cfg.freeze()
    return cfg


def _setup_distribution(
    a: argparse.Namespace,
    cfg: CfgNode,
) -> EmpiricalDistribution | MarchenkoPastur:
    """Define the distribution to be used.

    Returns
    -------
    EmpiricalDistribution | MarchenkoPastur
        The distribution instance.
    """
    if a.analytic:
        return MarchenkoPastur(ratio=cfg.DIST.RATIO, var=cfg.DIST.VAR)
    return load_data(cfg)


def _save_results(
    cfg: CfgNode,
    a: argparse.Namespace,
    evl: Float[np.ndarray, n_dof] | None,
    x: Float[np.ndarray, 5000],
    dimu2: Float[np.ndarray, 5000],
    dimu4: Float[np.ndarray, 5000],
    dimu6: Float[np.ndarray, 5000],
    dist: EmpiricalDistribution | MarchenkoPastur,
    logger: Logger,
) -> None:
    """Save the results to a JSON file."""
    output_dir: Path = Path(os.path.expandvars(cfg.DATA.OUTPUT_DIR)).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix: str = f"snr={cfg.SIG.SNR}"
    suffix_keys = set(a.suffix or [])
    mapping = {
        "nsamples": ("NUM_SAMPLES", "nsamples"),
        "ratio": ("RATIO", "ratio"),
        "seed": ("SEED", "seed"),
        "var": ("VAR", "var"),
        "lam": ("POIS_LAM", "lam"),
    }
    for key, (cfg_attr, label) in mapping.items():
        if key in suffix_keys:
            suffix += f"_{label}={getattr(cfg.DIST, cfg_attr)}"

    if a.analytic:
        suffix = f"analytic_var={cfg.DIST.VAR}_ratio={cfg.DIST.RATIO}_seed={cfg.DIST.SEED}"

    output_file: Path = output_dir / f"mp_canonical_dimensions_{suffix}.json"
    payload: dict[str, list[float] | float | None] = {
        "k2": x.tolist(),
        "evl": evl.tolist() if evl is not None else None,
        "dimu2": dimu2.tolist(),
        "dimu4": dimu4.tolist(),
        "dimu6": dimu6.tolist(),
        "dist": dist.ipdf(x).tolist(),
        "m2": dist.m2,
    }
    m2_mp: float | None = getattr(dist, "m2_mp", None)
    if m2_mp is not None:
        payload["m2_mp"]: float = float(m2_mp)
    with Path(output_file).open("w") as f:
        json.dump(payload, f)
    logger.info("Results saved in %s" % output_file)


def main(argv: list[str] | None = None) -> int | str:
    """Run the canonical dimensions computation script.

    Parameters
    ----------
    argv : list[str], optional
        The command-line arguments.

    Returns
    -------
    int | str
        The exit code or an error message.
    """
    a: argparse.Namespace = _parse_args(argv)

    # Get the logger
    logger_level: int = 10 * (4 - a.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")
    cfg: CfgNode = _load_config(a, logger)

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    logger.info("Computing the canonical dimensions...")

    # Define the distribution
    dist: EmpiricalDistribution | MarchenkoPastur = _setup_distribution(a, cfg)

    # Distribution parameters
    x_max: float = cfg.POT.UV_SCALE
    n_vars: int = int(cfg.DIST.NUM_SAMPLES * cfg.DIST.RATIO)
    x_min: float = 0.0 if a.analytic else 1.0 / np.sqrt(n_vars)

    # Compute the canonical dimensions
    x: Float[np.ndarray, 5000] = np.linspace(x_min, x_max, num=5000)
    dimu2: Float[np.ndarray, 5000]
    dimu4: Float[np.ndarray, 5000]
    dimu6: Float[np.ndarray, 5000]
    dimu2, dimu4, dimu6, _ = dist.canonical_dimensions(x).T

    # Save data
    evl: Float[np.ndarray, n_dof] | None = getattr(dist, "eigenvalues_", None)
    _save_results(cfg, a, evl, x, dimu2, dimu4, dimu6, dist, logger)

    return 0


def cli():  # noqa
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

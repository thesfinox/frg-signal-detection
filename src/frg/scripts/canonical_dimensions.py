#! /usr/bin/env python3
"""
Canonical Dimensions
--------------------

Compute the canonical dimensions of the couplings in a theory with given momenta distribution.

Authors
-------

- Riccardo Finotello <riccardo.finotello@cea.fr>
- Parham Radpay <parhamradpay@gmail.com>

Maintainer
----------

- Riccardo Finotello
"""

from __future__ import annotations

import argparse
import json
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

__author__: str = "Riccardo Finotello and Parham Radpay"
__email__: str = "riccardo.finotello@cea.fr; parham.radpay@gmail.com"
__description__: str = "Compute the canonical dimensions of the couplings in a theory with given momenta distribution."
__epilog__: str = (
    "For bug reports and info: " + __author__ + " <" + __email__ + ">"
)


def main(argv: list[str] | None = None) -> int | str:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("--config", required=False, help="Configuration file")
    parser.add_argument(
        "--analytic", action="store_true", help="Run an analytic simulation"
    )
    parser.add_argument(
        "--print_config", action="store_true", help="Print configuration"
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
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    a: argparse.Namespace = parser.parse_args(argv)

    # Get the logger
    logger_level: int = 10 * (4 - a.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")
    cfg: CfgNode = get_cfg_defaults()

    # Open the configuration file
    if a.config is None:
        logger.debug("No configuration file specified")
    else:
        logger.debug("Configuration file: %s" % a.config)
        cfg_file = Path(a.config)
        if cfg_file.exists():
            logger.debug("Configuration file exists!")
            cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(a.args)
    cfg.freeze()

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    logger.info("Computing the canonical dimensions...")

    # Distribution parameters
    x_max: float = cfg.POT.UV_SCALE
    n_vars: int = int(cfg.DIST.NUM_SAMPLES * cfg.DIST.RATIO)
    x_min: float = 1.0 / np.sqrt(n_vars)  # the smallest bin

    # Define the distribution
    if a.analytic:
        x_min: float = 0.0  # analytic can go to zero
        dist: MarchenkoPastur = MarchenkoPastur(
            ratio=cfg.DIST.RATIO, var=cfg.DIST.VAR
        )
    else:
        dist: EmpiricalDistribution = load_data(cfg)

    # Compute the canonical dimensions
    x: Float[np.ndarray, "5000"] = np.linspace(x_min, x_max, num=5000)
    dimu2: Float[np.ndarray, "5000"]
    dimu4: Float[np.ndarray, "5000"]
    dimu6: Float[np.ndarray, "5000"]
    dimu2, dimu4, dimu6, _ = dist.canonical_dimensions(x).T

    # Save data
    output_dir: Path = Path(cfg.DATA.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix: str = f"snr={cfg.SIG.SNR}"
    suffix_keys = set(a.suffix or [])
    if "nsamples" in suffix_keys:
        suffix += f"_nsamples={cfg.DIST.NUM_SAMPLES}"
    if "ratio" in suffix_keys:
        suffix += f"_ratio={cfg.DIST.RATIO}"
    if "seed" in suffix_keys:
        suffix += f"_seed={cfg.DIST.SEED}"
    if "var" in suffix_keys:
        suffix += f"_var={cfg.DIST.VAR}"
    if "lam" in suffix_keys:
        suffix += f"_lam={cfg.DIST.POIS_LAM}"
    if a.analytic:
        suffix = f"analytic_var={cfg.DIST.VAR}_ratio={cfg.DIST.RATIO}_seed={cfg.DIST.SEED}"
    output_file: Path = output_dir / f"mp_canonical_dimensions_{suffix}.json"
    payload: dict[str, list[float] | float] = {
        "k2": x.tolist(),
        "dimu2": dimu2.tolist(),
        "dimu4": dimu4.tolist(),
        "dimu6": dimu6.tolist(),
        "dist": dist.ipdf(x).tolist(),
        "m2": dist.m2,
    }
    m2_mp: float | None = getattr(dist, "m2_mp", None)
    if m2_mp is not None:
        payload["m2_mp"]: float = float(m2_mp)
    with open(output_file, "w") as f:
        json.dump(payload, f)
    logger.info("Results saved in %s" % output_file)

    return 0


def cli():
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
    cli()

#! /usr/bin/env python3
"""Study the distribution of the eigenvectors at different levels of signal-to-noise ratio."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from frg import EmpiricalDistribution, get_cfg_defaults, get_logger, load_data

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float
    from yacs.config import CfgNode

    # Dummy variables for jaxtyping to prevent Ruff F821 errors.
    # Defined strictly within TYPE_CHECKING so they cannot exist at runtime,
    # ensuring they never overwrite or conflict with actual code variables.
    p: int = 500


__author__: str = "Riccardo Finotello"
__email__: str = "riccardo.finotello@cea.fr"
__description__: str = "Study the distribution of the eigenvectors at different levels of signal-to-noise ratio."
__epilog__: str = (
    "For bug reports and info: " + __author__ + " <" + __email__ + ">"
)


def main(argv: list[str] | None = None) -> int | str:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__,
        epilog=__epilog__,
    )
    parser.add_argument("--config", required=False, help="Configuration file")
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print configuration",
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

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    logger.info("Computing the distribution of the eigenvectors...")
    dist: EmpiricalDistribution = load_data(cfg)
    evl: Float[np.ndarray, p] = dist.eigenvalues
    if dist.eigenvectors_ is not None:
        evc: Float[np.ndarray, p, p] = dist.eigenvectors_

    # Save the distribution of the eigenvectors
    output_dir: Path = Path(os.path.expandvars(cfg.DATA.OUTPUT_DIR)).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file: Path = (
        output_dir / f"mp_evc_distribution_snr={cfg.SIG.SNR}.json"
    )
    payload: dict[str, list[float] | float] = {
        "evl": evl.tolist(),
        "evc": evc.tolist(),
        "lplus": dist.lplus,
        "lplus_mp": dist.lplus_mp,
    }
    m2_mp: float | None = getattr(dist, "m2_mp", None)
    if m2_mp is not None:
        payload["m2_mp"]: float = m2_mp
    with Path(output_file).open("w") as f:
        json.dump(payload, f)
    logger.info("Results saved in %s" % output_file)

    return 0


def cli():  # noqa
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

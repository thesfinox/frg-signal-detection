#! /usr/bin/env python3
"""
FRG Equations
-------------

Functional Renormalization Group equations (LPA)

Compute the running of the couplings in a theory with given momenta distribution. Use the Local Potential Approximation.

Authors
-------

- Riccardo Finotello <riccardo.finotello@cea.fr>

Maintainers
-----------

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

    # Dummy variables for jaxtyping to prevent Ruff F821 errors.
    # Defined strictly within TYPE_CHECKING so they cannot exist at runtime,
    # ensuring they never overwrite or conflict with actual code variables.
    S: int = 100

__author__: str = "Riccardo Finotello"
__email__: str = "riccardo.finotello@cea.fr"
__description__: str = "Compute the running of the couplings in a theory with given momenta distribution. Use the Local Potential Approximation."
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
        else:
            logger.error("Configuration file %s does not exist!", a.config)
            raise FileNotFoundError(
                "Configuration file %s does not exist!" % a.config
            )
    cfg.merge_from_list(a.args)
    cfg.freeze()

    if a.print_config:
        print(cfg.dump())
        return 0

    # Run the simulation
    logger.info("Computing the running coupling...")

    # Distribution parameters
    x_uv: float = cfg.POT.UV_SCALE
    x_ir: float = float(
        1 / np.sqrt(cfg.DIST.NUM_SAMPLES)
    )  # stop at physical scale

    # Define the distribution
    if a.analytic:
        x_ir: float = 0.0  # analytic can go to zero
        dist: MarchenkoPastur = MarchenkoPastur(
            ratio=cfg.DIST.RATIO, var=cfg.DIST.VAR
        )
    else:
        dist: EmpiricalDistribution = load_data(cfg)

    # Compute the running
    k2: Float[np.ndarray, "S"]
    kappa: Float[np.ndarray, "S"]
    u4: Float[np.ndarray, "S"]
    u6: Float[np.ndarray, "S"]
    k2, kappa, u4, u6 = dist.frg_equations_lpa(
        x_uv,
        kappa_init=cfg.POT.KAPPA_INIT,
        u4_init=cfg.POT.U4_INIT,
        u6_init=cfg.POT.U6_INIT,
        dx=0.1 / cfg.DIST.NUM_SAMPLES,
        x_ir=x_ir,
    ).T

    # Save data
    output_dir: Path = Path(cfg.DATA.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file: Path = (
        output_dir
        / f"mp_frg_equations_lpa_snr={cfg.SIG.SNR}_kappa={cfg.POT.KAPPA_INIT}_u4={cfg.POT.U4_INIT}_u6={cfg.POT.U6_INIT}.json"
    )
    payload: dict[str, list[float] | float] = {
        "k2": k2.tolist(),
        "kappa": kappa.tolist(),
        "u4": u4.tolist(),
        "u6": u6.tolist(),
        "m2": dist.m2,
    }
    m2_mp: float | None = getattr(dist, "m2_mp", None)
    if m2_mp is not None:
        payload["m2_mp"] = m2_mp
    with open(output_file, "w") as f:
        json.dump(payload, f)
    logger.info("Data saved in %s" % output_file)

    return 0


def cli():
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

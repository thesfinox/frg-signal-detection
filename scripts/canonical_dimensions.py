#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Compute the canonical dimensions of the couplings in a theory with given momenta distribution.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from frg import MarchenkoPastur, get_cfg_defaults, get_logger
from frg.utils.utils import load_data

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Compute the canonical dimensions of the couplings in a theory with given momenta distribution."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")
    cfg = get_cfg_defaults()

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
    x_max = cfg.POT.UV_SCALE
    x_min = 1.0 / np.sqrt(cfg.DIST.NUM_SAMPLES)  # the smallest bin

    # Define the distribution
    if a.analytic:
        x_min = 0.0  # analytic can go to zero
        dist = MarchenkoPastur(ratio=cfg.DIST.RATIO, sigma=cfg.DIST.SIGMA)
    else:
        dist = load_data(cfg)

    # Compute the canonical dimensions
    x = np.linspace(x_min, x_max, num=1000)
    dimu2, dimu4, dimu6, _ = dist.canonical_dimensions(x).T

    # Save data
    output_dir = Path(cfg.DATA.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "analytic" if a.analytic else f"snr={cfg.SIG.SNR}"
    output_file = output_dir / f"mp_canonical_dimensions_{suffix}.json"
    payload = {
        "k2": x.tolist(),
        "dimu2": dimu2.tolist(),
        "dimu4": dimu4.tolist(),
        "dimu6": dimu6.tolist(),
        "dist": dist.ipdf(x).tolist(),
    }
    with open(output_file, "w") as f:
        json.dump(payload, f)
    logger.info("Results saved in %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
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
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

#! /usr/bin/env python3
"""
Plot eigenvectors

Plot the distribution of the eigenvectors at different levels of signal-to-noise ratio.
"""

import argparse
import json
import sys
from pathlib import Path

from frg.utils.utils import get_cfg_defaults, get_logger, load_data

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Plot the distribution of the eigenvectors at different levels of signal-to-noise ratio."
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
    logger.info("Computing the distribution of the eigenvectors...")
    dist = load_data(cfg)
    evl = dist.eigenvalues  # compute eigenvalues
    evc = dist.eigenvectors_  # compute eigenvectors

    # Find the spikes
    logger.debug("Computing the position of spikes...")
    spikes = dist.find_spikes(evl)
    evl = evl[:spikes]
    evc = evc[:, :spikes]

    # Compute the momenta
    k2 = 1 / evl
    k2 -= k2.min()

    # Save the distribution of the eigenvectors
    output_dir = Path(cfg.DATA.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"mp_evc_distribution_snr={cfg.SIG.SNR}.json"
    payload = {"k2": k2.tolist(), "evc": evc.tolist()}
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

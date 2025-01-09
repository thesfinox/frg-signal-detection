#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Compute the running of the couplings in a theory with given momenta distribution. Use the Local Potential Approximation.
"""

import argparse
import json
import sys
from pathlib import Path

from frg import MarchenkoPastur, get_cfg_defaults, get_logger

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Compute the running of the couplings in a theory with given momenta distribution. Use the Local Potential Approximation."
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
    if a.analytic:
        logger.info(
            "Computing the running coupling of the analytic distribution..."
        )

        # Marchenko-Pastur distribution
        x_uv = cfg.DIST.UV_SCALE
        mp = MarchenkoPastur(ratio=cfg.DIST.RATIO, sigma=cfg.DIST.SIGMA)
        k2, kappa, u4, u6 = mp.frg_equations_lpa(
            x_uv,
            kappa_init=cfg.POT.KAPPA_INIT,
            u4_init=cfg.POT.U4_INIT,
            u6_init=cfg.POT.U6_INIT,
        ).T

        # Save data
        output_dir = Path(cfg.DATA.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir
            / f"mp_frg_equations_lpa_kappa={cfg.POT.KAPPA_INIT}_u4={cfg.POT.U4_INIT}_u6={cfg.POT.U6_INIT}.json"
        )
        payload = {
            "k2": k2.tolist(),
            "kappa": kappa.tolist(),
            "u4": u4.tolist(),
            "u6": u6.tolist(),
        }
        with open(output_file, "w") as f:
            json.dump(payload, f)
        logger.info("Data saved in %s" % output_file)

        return 0

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
        help="Additional configuration arguments",
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

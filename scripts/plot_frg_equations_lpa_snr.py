#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Plot the dimension of the region containing symmetric points in the IR, using the Local Potential Approximation.
"""

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from frg import get_logger

mpl.use("agg")
plt.style.use("grayscale")

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = (
    "Plot the dimension of the region containing symmetric points in the IR."
)
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Load data
    phase = []
    snr = []
    symmetric = []
    broken = []
    for d in a.data:
        # Check if file exists
        d = Path(d)
        if not d.exists():
            logger.error("Data file %s does not exist!", d)
            raise FileNotFoundError("Data file %s does not exist!" % d)
        logger.debug("Opening data file %s" % d)
        with open(str(d)) as f:
            data = json.load(f)

        # Parse the content of the file (stop at the chosen scale)
        k2 = np.array(data["k2"])
        idx = np.argmin(np.abs(k2 - a.scale))
        snr.append(
            float(
                [i for i in d.stem.split("_") if "snr" in i][0].split("=")[-1]
            )
        )
        phase.append(True if np.abs(data["kappa"][idx]) < 1.0e-6 else False)
    snr = np.array(snr)
    phase = np.array(phase)

    # Group by SNR
    groups = groupby(zip(snr, phase), key=lambda x: x[0])
    groups = [(float(k), [v[1] for v in v]) for k, v in groups]
    symmetric = np.array([(k, float(sum(v))) for k, v in groups])
    broken = np.array([(k, len(v) - float(sum(v))) for k, v in groups])

    # Plot the results
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.plot(
        symmetric[..., 0],
        symmetric[..., 1] / (symmetric[..., 1] + broken[..., 1]),
        "bx",
        label="symmetric",
        alpha=0.85,
    )
    ax.plot(
        symmetric[..., 0],
        symmetric[..., 1] / (symmetric[..., 1] + broken[..., 1]),
        "b-",
        alpha=0.15,
    )

    ax.plot(
        broken[..., 0],
        broken[..., 1] / (symmetric[..., 1] + broken[..., 1]),
        "ro",
        label="broken",
        alpha=0.85,
    )
    ax.plot(
        broken[..., 0],
        broken[..., 1] / (symmetric[..., 1] + broken[..., 1]),
        "r-",
        alpha=0.15,
    )

    ax.set(xlabel=r"signal-to-noise ratio ($\beta$)", ylabel=r"fraction")
    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))

    # Save the file
    output_file = Path(a.output)
    output_dir = Path(output_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("Plot saved to %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("data", nargs="+", help="Data file(s) in JSON format")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="The momentum scale at which to compute the canonical dimensions",
    )
    parser.add_argument(
        "--output", default="mp_frg_equations.png", help="Output file"
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

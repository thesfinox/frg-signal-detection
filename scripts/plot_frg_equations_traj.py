#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Plot the running coupling in a theory with given momenta distribution.
"""

import argparse
import json
import sys
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
    "Plot the running coupling in a theory with given momenta distribution."
)
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Load data
    fig, ax = plt.subplots(
        ncols=3,
        figsize=(21, 5),
        layout="constrained",
    )
    for d in a.data:
        # Check if file exists
        d = Path(d)
        if not d.exists():
            logger.error("Data file %s does not exist!", d)
            raise FileNotFoundError("Data file %s does not exist!" % d)
        logger.debug("Opening data file %s" % d)
        with open(str(d)) as f:
            data = json.load(f)

        # Parse the content of the file
        u2 = data["u2"]
        u4 = data["u4"]
        u6 = data["u6"]

        # Reduce the number of points according to the chosen scale
        k2 = np.array(data["k2"])
        idx = np.argmin(np.abs(k2 - a.scale))
        u2 = u2[:idx]
        u4 = u4[:idx]
        u6 = u6[:idx]

        # Plot the dimensions
        logger.debug("Plotting the running coupling in file %s..." % d)
        ax[0].axvline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[0].axhline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[0].plot(u2, u4, "k-")
        ax[0].plot([u2[0]], [u4[0]], "bo")
        ax[0].plot([u2[-1]], [u4[-1]], "ro")
        ax[0].set(xlabel=r"$u_{2}$", ylabel=r"$u_{4}$")
        ax[0].ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )

        ax[1].axvline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[1].axhline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[1].plot(u2, u6, "k-")
        ax[1].plot([u2[0]], [u6[0]], "bo")
        ax[1].plot([u2[-1]], [u6[-1]], "ro")
        ax[1].set(xlabel=r"$u_{2}$", ylabel=r"$u_{6}$")
        ax[1].ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )

        ax[2].axvline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[2].axhline(0.0, color="k", linestyle="--", alpha=0.15)
        ax[2].plot(u4, u6, "k-")
        ax[2].plot([u4[0]], [u6[0]], "bo")
        ax[2].plot([u4[-1]], [u6[-1]], "ro")
        ax[2].set(xlabel=r"$u_{4}$", ylabel=r"$u_{6}$")
        ax[2].ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )

    # Save the file
    output_file = Path(a.output)
    output_dir = Path(output_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("Running couplings saved to %s" % output_file)

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

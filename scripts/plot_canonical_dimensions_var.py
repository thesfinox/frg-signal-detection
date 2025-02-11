#! /usr/bin/env python3
"""
Plot VAR response

Plot the canonical dimensions at a given scale for different values of the variance.
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
__description__ = "Plot the canonical dimensions at a given scale for different values of the variance."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Load file
    var = []
    dimu2 = []
    dimu4 = []
    dimu6 = []
    for f in a.results:
        # Check if the file exists
        f = Path(f)
        if not f.exists():
            logger.error("Data file %s does not exist!", f)
            raise FileNotFoundError("Data file %s does not exist!" % f)

        # Open the file
        logger.debug("Opening data file %s" % f)
        with open(str(f)) as file:
            data = json.load(file)

        # Find the value of the variance
        name = f.stem.split("_")
        name = [i for i in name if "var" in i][0]
        var.append(float(name.split("=")[-1]))

        # Parse the content of the file
        k2 = np.array(data["k2"])
        idx = np.argmin(np.abs(k2 - a.scale))
        dimu2.append(data["dimu2"][idx])
        dimu4.append(data["dimu4"][idx])
        dimu6.append(data["dimu6"][idx])

    # Plot the results
    logger.debug("Plotting the canonical dimensions...")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    sorting_order = np.argsort(var)
    var = np.array(var)[sorting_order]
    dimu2 = np.array(dimu2)[sorting_order]
    dimu4 = np.array(dimu4)[sorting_order]
    dimu6 = np.array(dimu6)[sorting_order]
    ax.plot(var, dimu2, "r-", alpha=0.15)
    ax.plot(var, dimu4, "g-", alpha=0.15)
    ax.plot(var, dimu6, "b-", alpha=0.15)
    ax.plot(var, dimu2, "rx", label=r"$\text{dim}(u_{2})$", alpha=0.85)
    ax.plot(var, dimu4, "gx", label=r"$\text{dim}(u_{4})$", alpha=0.85)
    ax.plot(var, dimu6, "bx", label=r"$\text{dim}(u_{6})$", alpha=0.85)
    ax.set(xlabel=r"variance", ylabel="canonical dimension")
    ax.ticklabel_format(
        style="sci", axis="both", scilimits=(0, 0), useMathText=True
    )
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))

    # Save the file
    output_file = Path(a.output)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("Variance plot saved to %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("results", type=str, nargs="+", help="Results files")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="The momentum scale at which to compute the canonical dimensions",
    )
    parser.add_argument(
        "--output",
        default="mp_canonical_dimensions_var.png",
        help="Output file",
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

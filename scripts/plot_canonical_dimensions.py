#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Plot the canonical dimensions of the couplings in a theory with given momenta distribution.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt

from frg import get_logger

mpl.use("agg")
plt.style.use("grayscale")


__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = "Plot the canonical dimensions of the couplings in a theory with given momenta distribution."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Load file
    data = Path(a.data)
    if not data.exists():
        logger.error("Data file %s does not exist!", a.data)
        raise FileNotFoundError("Data file %s does not exist!" % a.data)
    logger.debug("Opening data file %s")
    with open(str(data)) as f:
        data = json.load(f)

    # Parse the content of the file
    x = data["k2"]
    dimu2 = data["dimu2"]
    dimu4 = data["dimu4"]
    dimu6 = data["dimu6"]
    dist = data["dist"]

    # Plot the dimensions
    logger.debug("Plotting the canonical dimensions...")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.plot(x, dimu2, "r-", label=r"$\text{dim}(u_{2})$")
    ax.plot(x, dimu4, "g-", label=r"$\text{dim}(u_{4})$")
    ax.plot(x, dimu6, "b-", label=r"$\text{dim}(u_{6})$")
    ax.set(xlabel=r"$k^2$", ylabel="canonical dimension")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )
    ax2 = ax.twinx()
    ax2.plot(x, dist, "k--")
    ax2.set(ylabel="PDF")

    # Save the file
    output_file = Path(a.output)
    output_dir = Path(output_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("Canonical dimensions saved to %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("data", help="Data file in JSON format")
    parser.add_argument(
        "--output", default="mp_canonical_dimensions.png", help="Output file"
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

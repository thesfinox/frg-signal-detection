#! /usr/bin/env python3
"""
Plot SNR response

Plot the eigenvector distribution as a function of the SNR.
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
__description__ = "Plot the eigenvector distribution as a function of the SNR."
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(a: argparse.Namespace) -> int | str:
    # Get the logger
    logger_level = 10 * (4 - a.verb)
    logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Load file
    snr = []
    mode = []
    width = []
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

        # Find the value of the SNR
        name = f.stem.split("_")
        name = [i for i in name if "snr" in i][0]
        snr.append(float(name.split("=")[-1]))

        # Parse the content of the file
        k2 = np.array(data["k2"])
        evc = np.array(data["evc"])
        idx = np.argmin(np.abs(k2 - a.scale))

        # Define the boundaries of the "signal"
        sig = (
            max(0, idx - int(0.01 * len(k2))),
            min(idx + int(0.01 * len(k2)), len(k2)),
        )
        evc = evc[:, sig[0] : sig[1]]
        evc /= np.linalg.norm(evc, axis=0)  # normalization
        evc = evc.ravel()
        width.append(float(evc.std()))

        # Compute the histogram
        hist, bins = np.histogram(evc, bins=a.nbins, density=True)
        dx = np.diff(bins)
        mode_idx = np.argmax(hist)
        mode.append(float(bins[mode_idx] + dx[mode_idx] / 2.0))

    # Plot the results
    logger.debug("Plotting the values of the eigenvector distribution...")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    idx = np.argsort(np.array(snr))
    snr = np.array(snr)[idx]
    mode = np.array(mode)[idx]
    width = np.array(width)[idx]
    mode_plot = ax.plot(snr, mode, "k-", label="mode")
    ax.set(xlabel=r"signal-to-noise ratio ($\beta$)", ylabel="mode")
    ax.ticklabel_format(
        style="sci", axis="both", scilimits=(0, 0), useMathText=True
    )
    ax2 = ax.twinx()
    width_plot = ax2.plot(snr, width, "r--", label="width")
    ax2.set(xlabel=r"signal-to-noise ratio ($\beta$)", ylabel="width")
    ax2.ticklabel_format(
        style="sci", axis="both", scilimits=(0, 0), useMathText=True
    )
    ax.legend(
        handles=[mode_plot[0], width_plot[0]],
        labels=["mode", "width"],
        loc="lower left",
        bbox_to_anchor=(1.1, 0.0),
    )

    # Save the file
    output_file = Path(a.output)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("SNR plot saved to %s" % output_file)

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
        default="mp_evc_distribution_snr.png",
        help="Output file",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=1000,
        help="Number of bins in the histogram",
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Plot the distribution of the eigenvector components.
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
__description__ = "Plot the distribution of the eigenvector components."
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
    logger.debug("Opening data file %s" % a.data)
    with open(str(data)) as f:
        data = json.load(f)

    # Parse the content of the file
    k2 = np.array(data["k2"])
    evc = np.array(data["evc"])

    # Select a sample of signal and a sample of noise
    idx_sig = np.argmin(np.abs(k2 - a.scale))

    # Define the boundaries of the intervals of "signal" and "bkg"
    sig = (
        max(0, idx_sig - int(0.01 * len(k2))),
        min(idx_sig + int(0.01 * len(k2)), len(k2)),
    )
    idx_bkg = int(sig[1] - sig[0])
    evc_sig = evc[:, sig[0] : sig[1]]
    evc_sig /= np.linalg.norm(evc_sig, axis=0)  # normalization
    evc_sig_first = evc_sig[0]
    evc_sig = evc_sig.ravel()
    evc_bkg = evc[:, :idx_bkg]
    evc_bkg /= np.linalg.norm(evc_bkg, axis=0)  # normalization
    evc_bkg_first = evc_bkg[0]
    evc_bkg = evc_bkg.ravel()

    # Plot the distributions
    logger.debug("Plotting the distributions of the eigenvectors...")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    _, bins_first, _ = ax.hist(
        evc_bkg_first,
        histtype="step",
        density=True,
        label="background (1st comp.)",
        color="k",
        alpha=0.35,
    )
    n, bins, bkg_patches = ax.hist(
        evc_bkg,
        bins=a.nbins,
        histtype="step",
        density=True,
        label="background",
        color="k",
    )
    dx = np.diff(bins)
    mode_idx = np.argmax(n)
    mode_bkg = bins[mode_idx] + dx[mode_idx] / 2.0
    ax.axvline(mode_bkg, 0.0, max(n), color="k", ls="--", alpha=0.5)
    ax.hist(
        evc_sig_first,
        bins=bins_first,
        histtype="step",
        density=True,
        label="signal (1st comp.)",
        color="r",
        alpha=0.35,
    )
    n, bins, sig_patches = ax.hist(
        evc_sig,
        bins=bins,
        histtype="step",
        density=True,
        label="signal",
        color="r",
    )
    dx = np.diff(bins)
    mode_idx = np.argmax(n)
    mode_sig = bins[mode_idx] + dx[mode_idx] / 2.0
    ax.axvline(mode_sig, 0.0, max(n), color="r", ls="--", alpha=0.5)
    ax.set(xlabel="components", ylabel="density")
    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.legend(
        handles=[bkg_patches[0], sig_patches[0]],
        labels=[
            f"background (mode: {mode_bkg:.4f})",
            f"signal (mode: {mode_sig:.4f})",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    # Save the file
    output_file = Path(a.output)
    output_dir = Path(output_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    logger.debug("Distributions saved to %s" % output_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument("data", help="Data file in JSON format")
    parser.add_argument(
        "--scale", type=float, default=0.01, help="Energy scale"
    )
    parser.add_argument(
        "--output", default="mp_canonical_dimensions.png", help="Output file"
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

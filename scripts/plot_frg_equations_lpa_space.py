#! /usr/bin/env python3
"""
Functional Renormalization Group for Signal Detection

Plot the phase space of the FRG flow: for each initial condition in the UV, we plot whether the trajectory ended in a symmetryc phase (u2 > 0) or not.
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
    coords = []
    phase = []
    colors = []
    labels = []
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
        coords.append([data["kappa"][0], data["u4"][0], data["u6"][0]])
        phase.append(True if np.abs(data["kappa"][-1]) < 1.0e-6 else False)
        colors.append("g" if np.abs(data["kappa"][-1]) < 1.0e-6 else "r")
        labels.append(
            "symmetric" if np.abs(data["kappa"][-1]) < 1.0e-6 else "broken"
        )
    coords = np.array(coords)
    phase = np.array(phase)

    # Plot the results
    fig, ax = plt.subplots(
        ncols=3,
        nrows=2,
        figsize=(21, 10),
        layout="constrained",
    )
    ax = ax.ravel()

    ax[0].axvline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[0].axhline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[0].scatter(
        coords[..., 0][phase],
        coords[..., 1][phase],
        color="g",
        marker="o",
        label="symmetric",
    )
    ax[0].scatter(
        coords[..., 0][~phase],
        coords[..., 1][~phase],
        color="r",
        marker="o",
        label="broken",
    )
    ax[0].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))
    ax[0].set(xlabel=r"$\kappa$", ylabel=r"$u_{4}$")
    ax[0].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    ax[1].axvline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[1].axhline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[1].scatter(
        coords[..., 0][phase],
        coords[..., 2][phase],
        color="g",
        marker="o",
        label="symmetric",
    )
    ax[1].scatter(
        coords[..., 0][~phase],
        coords[..., 2][~phase],
        color="r",
        marker="o",
        label="broken",
    )
    ax[1].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))
    ax[1].set(xlabel=r"$\kappa$", ylabel=r"$u_{6}$")
    ax[1].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    ax[2].axvline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[2].axhline(0.0, color="k", linestyle="--", alpha=0.15)
    ax[2].scatter(
        coords[..., 1][phase],
        coords[..., 2][phase],
        color="g",
        marker="o",
        label="symmetric",
    )
    ax[2].scatter(
        coords[..., 1][~phase],
        coords[..., 2][~phase],
        color="r",
        marker="o",
        label="broken",
    )
    ax[2].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0))
    ax[2].set(xlabel=r"$u_{4}$", ylabel=r"$u_{6}$")
    ax[2].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    # Create grid data
    phase_interp = 2.0 * phase.astype("float") - 1.0

    plot = ax[3].tricontourf(
        coords[..., 0],
        coords[..., 1],
        phase_interp,
        levels=1,
        colors=["r", "g"],
        alpha=0.5,
    )
    cbar = fig.colorbar(plot, ax=ax[3], fraction=0.045, pad=-0.25)
    cbar.set_ticks([-0.5, 0.5])
    cbar.set_ticklabels(["broken", "symmetric"])
    ax[3].set(xlabel=r"$\kappa$", ylabel=r"$u_{4}$")
    ax[3].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    plot = ax[4].tricontourf(
        coords[..., 0],
        coords[..., 2],
        phase_interp,
        levels=1,
        colors=["r", "g"],
        alpha=0.5,
    )
    cbar = fig.colorbar(plot, ax=ax[4], fraction=0.045, pad=-0.25)
    cbar.set_ticks([-0.5, 0.5])
    cbar.set_ticklabels(["broken", "symmetric"])
    ax[4].set(xlabel=r"$\kappa$", ylabel=r"$u_{6}$")
    ax[4].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    plot = ax[5].tricontourf(
        coords[..., 1],
        coords[..., 2],
        phase_interp,
        levels=1,
        colors=["r", "g"],
        alpha=0.5,
    )
    cbar = fig.colorbar(plot, ax=ax[5], fraction=0.045, pad=-0.25)
    cbar.set_ticks([-0.5, 0.5])
    cbar.set_ticklabels(["broken", "symmetric"])
    ax[5].set(xlabel=r"$u_{4}$", ylabel=r"$u_{6}$")
    ax[5].ticklabel_format(
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
        "--output", default="mp_frg_equations.png", help="Output file"
    )
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)

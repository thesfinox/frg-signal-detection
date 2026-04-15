#! /usr/bin/env python3
"""Plot the canonical dimensions as a function of the signal-to-noise ratio.

Display the behaviour of the canonical dimensions depending on the type of noise:

- **Gaussian-only noise:** display the canonical dimensions in a 1D plot as a function of the signal-to-noise ratio.
- **Gaussian and Poissonian noise:** display the canonical dimensions as a surface plot as a function of signal-to-noise ratio and the parameter of the Poissonian distribution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from frg import (
    canonical_dimensions_files,
    canonical_dimensions_files_poiss,
    get_logger,
    plot_canonical_dimensions,
    plot_canonical_dimensions_scan,
    plot_canonical_dimensions_scan2d,
)

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float

    n_steps: int = 1000

__author__ = "Riccardo Finotello"
__email__ = "riccardo.finotello@cea.fr"
__description__ = (
    "Plot the canonical dimensions as a function of the signal-to-noise ratio."
)
__epilog__ = "For bug reports and info: " + __author__ + " <" + __email__ + ">"


def main(argv: list[str] | None = None) -> int | str:
    """Run the canonical dimensions plotting script.

    Parameters
    ----------
    argv : list[str], optional
        The command-line arguments.

    Returns
    -------
    int | str
        The exit code or an error message.

    Raises
    ------
    ValueError
        If the input file or directory does not exist.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__,
        epilog=__epilog__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Directory containing the JSON files of JSON file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory of the plots",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for the output files",
    )
    parser.add_argument(
        "--analytic",
        action="store_true",
        help="Use analytic method for plotting",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image file for the output plot",
    )
    parser.add_argument(
        "-v",
        dest="verb",
        action="count",
        default=0,
        help="Verbosity level",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    # Get the logger
    logger_level: int = 10 * (4 - args.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    # Sanitise the input
    inp_dir_or_file: Path = Path(os.path.expandvars(args.input)).absolute()
    if not inp_dir_or_file.exists():
        logger.error("%s does not exist!", inp_dir_or_file)
        raise ValueError(f"{inp_dir_or_file} does not exist!")

    # If only one file, then plot the canonical dimensions at the precise value
    if inp_dir_or_file.is_file():
        logger.info(
            "Only one file found. Plotting canonical dimensions at the precise value...",
        )
        with Path(inp_dir_or_file).open() as f:
            data: dict[str, Float[np.ndarray, n_steps]] = json.load(f)
            plot_canonical_dimensions(
                data,
                suffix=args.suffix,
                output_dir=args.output,
                analytic=args.analytic,
            )
        return 0

    # If a directory, then plot the canonical dimensions as a function of the signal-to-noise ratio
    file_list: list[Path] = list(inp_dir_or_file.glob("*.json"))

    # Check if "lam=" is present in the file names (in which case, we need
    # to use 2D plots)
    is_poiss: bool = all("lam=" in f.name for f in file_list)

    # Plot the canonical dimensions
    snr: Float[np.ndarray, n_steps]
    dimu2: Float[np.ndarray, n_steps]
    dimu4: Float[np.ndarray, n_steps]
    dimu6: Float[np.ndarray, n_steps]
    if is_poiss:
        logger.info("Using 2D plots for Poissonian noise.")

        # Recover the informations
        lam: Float[np.ndarray, n_steps]
        snr, lam, dimu2, dimu4, dimu6 = canonical_dimensions_files_poiss(
            path=inp_dir_or_file,
        )

        # Plot the information
        plot_canonical_dimensions_scan2d(
            snr=snr,
            lam=lam,
            dimu2=dimu2,
            dimu4=dimu4,
            dimu6=dimu6,
            suffix=args.suffix,
            image=args.image,
            output_dir=args.output,
        )
    else:
        logger.info("Using 1D plots for Gaussian-only noise.")

        # Recover the informations
        snr, dimu2, dimu4, dimu6 = canonical_dimensions_files(
            path=inp_dir_or_file,
        )

        # Plot the information
        plot_canonical_dimensions_scan(
            x=snr,
            name="signal-to-noise ratio ($\\beta$)",
            dimu2=dimu2,
            dimu4=dimu4,
            dimu6=dimu6,
            win=40,
            suffix=args.suffix,
            image=args.image,
            output_dir=args.output,
        )

    return 0


def cli():  # noqa
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

#! /usr/bin/env python3
"""
Configuration Files
-------------------

Generate configurations files for the exploration of the phase space of the initial values of the couplings.

Authors
-------

- Riccardo Finotello <riccardo.finotello@cea.fr>

Maintainers
-----------

- Riccardo Finotello
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

from frg import get_cfg_defaults, get_logger

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float
    from matplotlib.axes import Axes
    from yacs.config import CfgNode

    # Dummy variables for jaxtyping to prevent Ruff F821 errors.
    # Defined strictly within TYPE_CHECKING so they cannot exist at runtime,
    # ensuring they never overwrite or conflict with actual code variables.
    n: int = 1000
    d: int = 2

__author__: str = "Riccardo Finotello"
__email__: str = "riccardo.finotello@cea.fr"
__description__: str = "Generate configurations files for the exploration of the phase space of the initial values of the couplings."
__epilog__: str = (
    "For bug reports and info: " + __author__ + " <" + __email__ + ">"
)


def main(argv: list[str] | None = None) -> int | str:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__, epilog=__epilog__
    )
    parser.add_argument(
        "--config", required=True, help="Base configuration file"
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Parameters to be sampled in JSON format. Keys must match nodes of the configuration file (case insensitive).",
    )
    parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
        help="Number of configurations to generate",
    )
    parser.add_argument(
        "--output_dir", default="configs", help="Output directory"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots to visualise sampling",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "-v", dest="verb", action="count", default=0, help="Verbosity level"
    )
    a: argparse.Namespace = parser.parse_args(argv)

    # Get the logger
    logger_level: int = 10 * (4 - a.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")
    cfg: CfgNode = get_cfg_defaults()

    # Open the configuration file
    if a.config is None:
        logger.debug("No configuration file specified")
    else:
        logger.debug("Configuration file: %s" % a.config)
        cfg_file = Path(a.config)
        if cfg_file.exists():
            logger.debug("Configuration file exists!")
            cfg.merge_from_file(cfg_file)
    cfg.freeze()

    # Output path
    output_dir: Path = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters to be samples
    params: Path = Path(a.params)
    if not params.exists():
        logger.error("Parameters file %s does not exist!", a.params)
        raise FileNotFoundError("Parameters file %s does not exist!" % a.params)
    logger.debug("Opening parameters file %s", a.params)
    with open(str(params)) as f:
        params: dict[str, dict[str, list[float]]] = json.load(f)

    # Generate the configurations
    names: list[tuple[str, str]] = []
    l_bounds: list[float] = []
    u_bounds: list[float] = []
    for key, value in params.items():
        for param, bounds in value.items():
            names.append((key.upper(), param.upper()))
            l_bounds.append(bounds[0])
            u_bounds.append(bounds[1])

    if len(names) > 1:
        logger.info("Generating LHS samples...")
        sampler: qmc.LatinHypercube = qmc.LatinHypercube(
            d=len(names), seed=a.seed
        )
        values: Float[np.ndarray, "n, d"] = sampler.random(n=a.n_samples)
        values: Float[np.ndarray, "n, d"] = qmc.scale(
            values, l_bounds, u_bounds
        )
    else:
        logger.info("Generating linear samples...")
        values: Float[np.ndarray, "n"] = np.linspace(
            l_bounds[0], u_bounds[0], num=a.n_samples
        )

    # Create the configurations
    for value in values:
        cfg_copy: CfgNode = cfg.clone()
        output_name: str = f"{cfg_file.stem}_"
        for i, name in enumerate(names):
            v: float = value if len(names) <= 1 else value[i]
            cfg_copy[name[0]][name[1]]: float = float(v)
            output_name += f"{name[1].lower()}={v:.9f}"
            if i < len(names) - 1:
                output_name += "_"
        cfg_copy.freeze()
        output_name += ".yaml"
        output_path: Path = output_dir / output_name
        with open(str(output_path), "w") as f:
            f.write(cfg_copy.dump())

    if a.plots:  # do not display unless explicitly requested
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        mpl.use("TkAgg")
        plt.style.use("grayscale")

        # Visualise sampling
        if len(names) > 1:
            comb: list[tuple[int, int]] = list(
                combinations(range(len(names)), 2)
            )
            for i, j in comb:
                x_label: str = ".".join(names[i])
                y_label: str = ".".join(names[j])
                x_values: Float[np.ndarray, "n"] = values[..., i]
                y_values: Float[np.ndarray, "n"] = values[..., j]

                _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
                ax.plot(x_values, y_values, "ko", alpha=0.5)
                ax.set(xlabel=x_label, ylabel=y_label)
                ax.ticklabel_format(
                    axis="both",
                    style="sci",
                    scilimits=(0, 0),
                    useMathText=True,
                )
                plt.show()
        else:
            x_label: str = ".".join(names[0])
            x_values: Float[np.ndarray, "n"] = values[..., 0]

            ax: Axes
            _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
            ax.plot(x_values, [0.0] * len(x_values), "kx", alpha=0.5)
            ax.set(xlabel=x_label, ylabel="", yticks=[])
            ax.ticklabel_format(
                axis="x",
                style="sci",
                scilimits=(0, 0),
                useMathText=True,
            )
            plt.show()

    return 0


def cli():
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
    cli()

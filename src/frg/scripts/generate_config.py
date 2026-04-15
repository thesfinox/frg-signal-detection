#! /usr/bin/env python3
"""Generate configurations files for the exploration of the phase space of the initial values of the couplings."""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

from frg import get_cfg_defaults, get_logger

if TYPE_CHECKING:
    from logging import Logger

    from jaxtyping import Float
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__description__,
        epilog=__epilog__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Base configuration file",
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
        "--output_dir",
        default="configs",
        help="Output directory",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots to visualise sampling",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "-v",
        dest="verb",
        action="count",
        default=0,
        help="Verbosity level",
    )
    return parser.parse_args(argv)


def _load_resources(
    a: argparse.Namespace,
    logger: Logger,
) -> tuple[CfgNode, Path, dict[str, dict[str, list[float]]], Path]:
    """Load configuration and parameters.

    Returns
    -------
    tuple[CfgNode, Path, dict, Path]
        The configuration node, configuration file path, parameters dictionary, and output directory path.

    Raises
    ------
    FileNotFoundError
        If the configuration or parameters file does not exist.
    """
    cfg: CfgNode = get_cfg_defaults()
    cfg_file = Path(os.path.expandvars(a.config)).absolute()
    if cfg_file.exists():
        cfg.merge_from_file(cfg_file)
    else:
        logger.error("Configuration file %s does not exist!", cfg_file)
        raise FileNotFoundError(
            "Configuration file %s does not exist!" % cfg_file,
        )
    cfg.freeze()

    output_dir: Path = Path(os.path.expandvars(a.output_dir)).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path: Path = Path(os.path.expandvars(a.params)).absolute()
    if not params_path.exists():
        logger.error("Parameters file %s does not exist!", params_path)
        raise FileNotFoundError(
            "Parameters file %s does not exist!" % params_path,
        )
    logger.debug("Opening parameters file %s", params_path)
    with Path(str(params_path)).open() as f:
        params: dict[str, dict[str, list[float]]] = json.load(f)

    return cfg, cfg_file, params, output_dir


def _parse_bounds(
    params: dict[str, dict[str, list[float]]],
) -> tuple[list[tuple[str, str]], list[float], list[float]]:
    """Parse parameters to extract names and bounds.

    Returns
    -------
    tuple[list[tuple[str, str]], list[float], list[float]]
        The list of parameter names, lower bounds, and upper bounds.
    """
    names: list[tuple[str, str]] = []
    l_bounds: list[float] = []
    u_bounds: list[float] = []
    for key, value in params.items():
        for param, bounds in value.items():
            names.append((key.upper(), param.upper()))
            l_bounds.append(bounds[0])
            u_bounds.append(bounds[1])
    return names, l_bounds, u_bounds


def _generate_samples(
    names: list[tuple[str, str]],
    l_bounds: list[float],
    u_bounds: list[float],
    n_samples: int,
    seed: int,
    logger: Logger,
) -> Float[np.ndarray, n, d] | Float[np.ndarray, n]:
    """Generate LHS or linear samples.

    Returns
    -------
    Float[np.ndarray, "n, d"] | Float[np.ndarray, "n"]
        The generated samples.
    """
    if len(names) > 1:
        logger.info("Generating LHS samples...")
        sampler: qmc.LatinHypercube = qmc.LatinHypercube(
            d=len(names),
            seed=seed,
        )
        values: Float[np.ndarray, n, d] = sampler.random(n=n_samples)
        return qmc.scale(values, l_bounds, u_bounds)

    logger.info("Generating linear samples...")
    return np.linspace(l_bounds[0], u_bounds[0], num=n_samples)


def _write_configs(
    values: np.ndarray,
    names: list[tuple[str, str]],
    cfg: CfgNode,
    cfg_file: Path,
    output_dir: Path,
) -> None:
    """Write the sampled configurations to files."""
    for value in values:
        cfg_copy: CfgNode = cfg.clone()
        output_name: str = f"{cfg_file.stem}_"
        for i, name in enumerate(names):
            v: float = float(value) if len(names) <= 1 else float(value[i])
            cfg_copy[name[0]][name[1]] = v
            output_name += f"{name[1].lower()}={v:.9f}"
            if i < len(names) - 1:
                output_name += "_"
        cfg_copy.freeze()
        output_name += ".yaml"
        output_path: Path = output_dir / output_name
        with Path(str(output_path)).open("w") as f:
            f.write(cfg_copy.dump())


def _visualize_sampling(
    values: np.ndarray,
    names: list[tuple[str, str]],
) -> None:
    """Visualize the sampling points."""
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("TkAgg")
    plt.style.use("grayscale")

    if len(names) > 1:
        comb: list[tuple[int, int]] = list(combinations(range(len(names)), 2))
        for i, j in comb:
            x_label: str = ".".join(names[i])
            y_label: str = ".".join(names[j])
            x_vals: Float[np.ndarray, n] = values[..., i]
            y_vals: Float[np.ndarray, n] = values[..., j]

            _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
            ax.plot(x_vals, y_vals, "ko", alpha=0.5)
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
        x_vals: Float[np.ndarray, n] = values[..., 0]

        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.plot(x_vals, [0.0] * len(x_vals), "kx", alpha=0.5)
        ax.set(xlabel=x_label, ylabel="", yticks=[])
        ax.ticklabel_format(
            axis="x",
            style="sci",
            scilimits=(0, 0),
            useMathText=True,
        )
        plt.show()


def main(argv: list[str] | None = None) -> int | str:
    """Run the configuration generation script.

    Parameters
    ----------
    argv : list[str], optional
        The command-line arguments.

    Returns
    -------
    int | str
        The exit code or an error message.
    """
    a: argparse.Namespace = _parse_args(argv)

    # Get the logger
    logger_level: int = 10 * (4 - a.verb)
    logger: Logger = get_logger(__name__, level=logger_level)
    logger.info("Starting...")

    cfg, cfg_file, params, output_dir = _load_resources(a, logger)
    names, l_bounds, u_bounds = _parse_bounds(params)
    values: np.ndarray = _generate_samples(
        names,
        l_bounds,
        u_bounds,
        a.n_samples,
        a.seed,
        logger,
    )

    # Create the configurations
    _write_configs(values, names, cfg, cfg_file, output_dir)

    if a.plots:
        _visualize_sampling(values, names)

    return 0


def cli():  # noqa
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

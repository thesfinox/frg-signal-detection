"""
Functions used for data analysis.

Author: Riccardo Finotello <riccardo.finotello@cea.fr>
Maintainers (name.surname@cea.fr) :

- Riccardo Finotello
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from PIL import Image

from frg.distributions.distributions import Distribution, MarchenkoPastur


def compute_roi(
    data: dict[str, Any],
    thresh: float = 0.5,
    analytic: bool = False,
) -> tuple[int, int, int]:
    """
    Compute the indices of the region of interest, and its initial and final points.

    Parameters
    ----------
    data : dict[str, Any]
        The results of the computation of the canonical dimensions.
    thresh : float
        The value of the threshold on the distribution to be considered "bulk". By default `0.5`.
    analytic : bool
        Treat the distribution as analytic. By default `False`.

    Returns
    -------
    tuple[int, int, int]
        The point of interest, the start of the region of interest, the end of the region of interest
    """
    top = np.argmax(data["dist"])
    if top == 0:
        return 0, 0, 0
    if analytic:
        return 1, 0, top
    start = np.argmin(np.abs(np.array(data["dist"][:top]) - thresh))

    idx = start + (top - start) // 2

    return int(idx), int(start), int(top)


def interp_canonical_dimensions(
    data: dict[str, Any], idx: int
) -> tuple[Any, Any, Any]:
    """
    Interpolate the behaviour of the canonical dimensions.

    Parameters
    ----------
    data : dict[str, Any]
        The experimental data.
    idx : int
        The index of the starting point.
    """
    stop = np.argmin(np.abs(np.array(data["k2"]) - 0.7))
    dimu2_interp = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu2"][idx:stop], 1)
    )
    dimu4_interp = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu4"][idx:stop], 1)
    )
    dimu6_interp = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu6"][idx:stop], 1)
    )
    return dimu2_interp, dimu4_interp, dimu6_interp


def extract_interp_values(
    data: dict[str, Any], thresh: float = 0.5, deep_ir: bool = False
) -> tuple[float, float, float, float]:
    """
    Extract the interpolated values.

    Parameters
    ----------
    data : dict[str, Any]
        The experimental data.
    thresh : float
        The value of the threshold on the distribution to be considered "bulk". By default `0.5`.
    deep_ir : bool
        Return the values of the interpolation at the deep IR scale. By default `False`.

    Returns
    -------
    tuple[float, float, float, float]
        The values of :math:`k^2`, :math:`\\text{dim}(u_{2})`, :math:`\\text{dim}(u_{4})`, and :math:`\\text{dim}(u_{6})` at the reference scale.
    """
    idx, _, _ = compute_roi(data, thresh, analytic=deep_ir)
    dimu2_interp, dimu4_interp, dimu6_interp = interp_canonical_dimensions(
        data, idx
    )

    k2 = float(data["k2"][idx]) if not deep_ir else 0.0
    return (
        k2,
        float(dimu2_interp(k2)),
        float(dimu4_interp(k2)),
        float(dimu6_interp(k2)),
    )


def canonical_dimensions_argsort(
    x: ArrayLike,
    dimu2: ArrayLike,
    dimu4: ArrayLike,
    dimu6: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Index sort the signal to noise ratio and the canonical dimensions.

    Parameters
    ----------
    x : ArrayLike
        The quantity of interest.
    dimu2 : ArrayLike
        The canonical dimension of the quadratic coupling.
    dimu4 : ArrayLike
        The canonical dimension of the quartic coupling.
    dimu6 : ArrayLike
        The canonical dimension of the sextic coupling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing the signal-to-noise ratio and the canonical dimensions in the same order as the input parameters.
    """
    x = np.array(x)
    dimu2 = np.array(dimu2)
    dimu4 = np.array(dimu4)
    dimu6 = np.array(dimu6)

    idx = np.argsort(x)

    return x[idx], dimu2[idx], dimu4[idx], dimu6[idx]


def canonical_dimensions_files(
    path: str, glob: str = "*.json", analytic: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Open multiple files and stores the canonical dimensions.

    Parameters
    ----------
    path : str
        The path to the directory containing the files.
    glob : str
        The global pattern to open. By default `"*.json"`.
    analytic : bool
        Treat the experiment as analytic. By default `False`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing the quantity of interest, and the canonical dimensions.
    """
    x, dimu2, dimu4, dimu6, scale = [], [], [], [], []

    files = Path(path).glob(glob)

    for file in files:
        with open(file) as f:
            data = json.load(f)
        add_values(
            extract_interp_values(data, deep_ir=analytic),
            scale,
            dimu2,
            dimu4,
            dimu6,
        )

        value = re.search("[0-9][.][0-9]*", file.stem)
        x.append(float(value.group()))

    return canonical_dimensions_argsort(x, dimu2, dimu4, dimu6)


def canonical_dimensions_ratio_files(
    path: str, glob: str = "*.json", analytic: bool = False
) -> pd.DataFrame:
    """
    Open multiple files as a function of ratio and seed and stores the canonical dimensions.

    Parameters
    ----------
    path : str
        The path to the directory containing the files.
    glob : str
        The global pattern to open. By default `"*.json"`.
    analytic : bool
        Treat the experiment as analytic. By default `False`.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the ratio, the seed, and the canonical dimensions.
    """
    ratio_l, seed_l, dimu2, dimu4, dimu6, scale = [], [], [], [], [], []

    files = Path(path).glob(glob)

    for file in files:
        with open(file) as f:
            data = json.load(f)
        add_values(
            extract_interp_values(data, deep_ir=analytic),
            scale,
            dimu2,
            dimu4,
            dimu6,
        )

        ratio = re.search("_ratio=[0-9][.][0-9]*?_", str(file)).group()[1:-1]
        ratio = float(ratio.split("=")[-1])
        ratio_l.append(ratio)

        seed = re.search("_seed=[0-9]*", str(file)).group()[1:]
        seed = int(seed.split("=")[-1])
        seed_l.append(seed)

    return pd.DataFrame(
        {
            "ratio": ratio_l,
            "seed": seed_l,
            "dimu2": dimu2,
            "dimu4": dimu4,
            "dimu6": dimu6,
        }
    )


def add_values(
    interp_values: tuple[float, float, float, float],
    scale: list[float],
    dimu2: list[float],
    dimu4: list[float],
    dimu6: list[float],
):
    """
    Add values to lists.

    Parameters
    ----------
    interp_values : tuple[float, float, float, float]
        The interpolated values of :math:`k^2`, :math:`\\text{dim}(u_{2})`, :math:`\\text{dim}(u_{4})`, :math:`\\text{dim}(u_{6})`.
    scale : list[float]
        The list of values of the reference scale :math:`k^2`.
    dimu2 : list[float]
        The list of values of :math:`\\text{dim}(u_{2})`.
    dimu4 : list[float]
        The list of values of :math:`\\text{dim}(u_{4})`.
    dimu6 : list[float]
        The list of values of :math:`\\text{dim}(u_{6})`.
    """
    k2, dimu2_value, dimu4_value, dimu6_value = interp_values
    scale.append(k2)
    dimu2.append(dimu2_value)
    dimu4.append(dimu4_value)
    dimu6.append(dimu6_value)


def plot_distribution(dist: Distribution, output_dir: str | Path = "plots"):
    """
    Plot distributions.

    Parameters
    ----------
    dist : Distribution
        The distribution to show.
    output_dir : str | Path
        The output directory of the plots. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    _, ax = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")

    # Show axes
    ax[0].axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[0].axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[0].set(xlabel="$\\lambda$", ylabel="$\\mu$")
    ax[1].axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[1].axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[1].set(xlabel="$k^2$", ylabel="$\\rho$")

    if isinstance(dist, MarchenkoPastur):
        # PDF
        x = np.linspace(0.0, 1.05 * dist.lplus, num=5000)
        y = dist.pdf(x)
        ax[0].plot(x, y, "k-")

        ax_inset = ax[0].inset_axes(
            [0.35, 1.55, 1.5, 1.5], transform=ax[0].transData
        )
        ax_inset.plot(x, y, "k-")
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(0.1 * dist.lminus, 20 * dist.lminus)
        ax_inset.set_title("UV")
        ax_inset.tick_params(labelsize=12)
        ax[0].indicate_inset_zoom(ax_inset, edgecolor="k")

        ax_inset = ax[0].inset_axes(
            [2.5, 0.75, 1.5, 1.5], transform=ax[0].transData
        )
        ax_inset.plot(x, y, "k-")
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(0.98 * dist.lplus, 1.01 * dist.lplus)
        ax_inset.set_ylim(-0.01, 0.05)
        ax_inset.set_title("IR")
        ax_inset.tick_params(labelsize=12)
        ax[0].indicate_inset_zoom(ax_inset, edgecolor="k")

        # PDF of the inverse
        x = np.linspace(0.0, 3.0, num=1000)
        y = dist.ipdf(x)
        ax[1].plot(x, y, "k-")

        ax_inset = ax[1].inset_axes(
            [0.95, 0.45, 1.15, 0.35], transform=ax[1].transData
        )
        ax_inset.plot(x, y, "k-")
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(-0.02, 0.35)
        ax_inset.set_title("IR")
        ax_inset.tick_params(labelsize=12)
        ax[1].indicate_inset_zoom(ax_inset, edgecolor="k")

        plt.savefig(
            output_dir
            / f"marchenkopastur_ratio={dist.ratio}_sigma={dist.sigma}.pdf"
        )
    else:
        evls = dist.eigenvalues_

        # PDF
        ax[0].hist(
            evls,
            bins=2 * int(np.sqrt(len(evls))),
            color="b",
            alpha=0.5,
            density=True,
        )

        x = np.linspace(0.0, 1.05 * dist.lplus, num=1000)
        y = dist.pdf(x)
        ax[0].plot(x, y, "k-")

        # PDF of the inverse
        x = np.linspace(0.0, 3.0, num=1000)
        y = dist.ipdf(x)
        ax[1].plot(x, y, "k-")

        plt.savefig(
            output_dir
            / f"empirical_dist_ratio={dist.ratio}_sigma={dist.sigma}_nsamples={dist.n_samples}.pdf"
        )


def plot_canonical_dimensions(
    data: dict[str, Any],
    thresh: float = 0.5,
    suffix: str | None = None,
    analytic: bool = False,
    output_dir: str | Path = "plots",
):
    """
    Plot a single instance of the canonical dimensions.

    Parameters
    ----------
    data : dict[str, Any]
        The results of the computation of the canonical dimensions.
    thresh : float
        The value of the threshold on the distribution to be considered "bulk". By default `0.5`.
    suffix : str, optional
        The suffix of the file name.
    analytic : bool
        Analytic computation. By default `False`.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    _, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    # Compute the point between the max and the start
    idx, start, top = compute_roi(data=data, thresh=thresh)

    ax.plot(
        data["k2"],
        data["dimu2"],
        "r-",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{2})$",
    )
    ax.plot(
        data["k2"],
        data["dimu4"],
        "g--",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{4})$",
    )
    ax.plot(
        data["k2"],
        data["dimu6"],
        "b-.",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{6})$",
    )

    # Interpolations
    if not analytic:
        dimu2_interp, dimu4_interp, dimu6_interp = interp_canonical_dimensions(
            data, idx
        )

        ax.plot(
            data["k2"],
            dimu2_interp(data["k2"]),
            "r-",
            label=r"$\text{dim}(u_{2})$",
        )
        ax.plot(
            data["k2"],
            dimu4_interp(data["k2"]),
            "g--",
            label=r"$\text{dim}(u_{4})$",
        )
        ax.plot(
            data["k2"],
            dimu6_interp(data["k2"]),
            "b-.",
            label=r"$\text{dim}(u_{6})$",
        )

        ax.axvspan(
            data["k2"][start],
            data["k2"][top],
            ls="dashed",
            color="r",
            alpha=0.05,
        )
        ax.axvline(data["k2"][idx], ls="dashed", color="r", alpha=0.25)

        ax.plot([data["k2"][idx]], [dimu2_interp(data["k2"][idx])], "ro")
        ax.text(
            x=data["k2"][idx] * 1.1,
            y=dimu2_interp(data["k2"][idx]),
            s=f"$\\text{{dim}}(u_2) = {dimu2_interp(data['k2'][idx]):.2f}$",
            color="r",
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.plot([data["k2"][idx]], [dimu4_interp(data["k2"][idx])], "go")
        ax.text(
            x=data["k2"][idx] * 1.1,
            y=dimu4_interp(data["k2"][idx]),
            s=f"$\\text{{dim}}(u_4) = {dimu4_interp(data['k2'][idx]):.2f}$",
            color="g",
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.plot([data["k2"][idx]], [dimu6_interp(data["k2"][idx])], "bo")
        ax.text(
            x=data["k2"][idx] * 1.1,
            y=dimu6_interp(data["k2"][idx]),
            s=f"$\\text{{dim}}(u_6) = {dimu6_interp(data['k2'][idx]):.2f}$",
            color="b",
            fontsize=10,
            ha="left",
            va="top",
        )

    ax.set_xlabel(r"$k^2$")
    ax.set_ylabel("canonical dimensions")
    ax.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )
    ax2 = ax.twinx()
    ax2.plot(data["k2"], data["dist"], "k--")
    ax2.set(ylabel="PDF")

    if suffix is None:
        plt.savefig(output_dir / "canonical_dimensions.pdf")
    else:
        plt.savefig(output_dir / f"canonical_dimensions_{suffix}.pdf")


def ema(x: ArrayLike, y: ArrayLike, win: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Exponential movin average (EMA).

    Parameters
    ----------
    x : ArrayLike
        The list of values.
    win : int
        The scaling width.

    Returns
    -------
    tuple[np.ndarray, lisy[float]]
        The smoothed list of x and y.
    """
    x, y = np.array(x), np.array(y)

    # Window
    win = np.ones((win,)) / win

    # Convolution
    new_x = np.convolve(x, win, mode="valid")
    new_y = np.convolve(y, win, mode="valid")

    return new_x, new_y


def plot_canonical_dimensions_scan(
    x: list[float],
    name: str,
    win: int = 0,
    dimu2: ArrayLike | None = None,
    dimu4: ArrayLike | None = None,
    dimu6: ArrayLike | None = None,
    suffix: str | None = None,
    image: str | Path = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the canonical dimensions as a function of a particular quantity of interest.

    Parameters
    ----------
    x : ArrayLike
        The quantity of interest.
    name : str
        The label of the x-axis.
    win : int
        The smoothing window width. By default `0`.
    dimu2 : ArrayLike, optional
        The list of values of the canonical dimension of the quadratic coupling.
    dimu4 : ArrayLike, optional
        The list of values of the canonical dimension of the quartic coupling.
    dimu6 : ArrayLike, optional
        The list of values of the canonical dimension of the sextic coupling.
    suffix : str, optional
        The suffix to postpone to the file name.
    image : str | Path, optional
        The path to the image used for the computations.
    output_dir : str | Path
        The output directory. By default `"plots"`.

    Raises
    ------
    FileNotFoundError
        If the provided image could not be found.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if image is not None:
        image = Path(image)
        if not image.exists():
            raise FileNotFoundError(
                f"The image you provided ({str(image)}) could not be found!"
            )
        image = np.array(Image.open(image))
        fig = plt.figure(figsize=(9, 5), layout="constrained")
        axs = fig.subplot_mosaic(
            [["left", "top_right"], ["left", "."]],
            width_ratios=[2, 1],
        )
        ax = axs["left"]  # compatibility with no image

        # Plot the image
        axs["top_right"].grid(False)
        axs["top_right"].axis("off")
        axs["top_right"].imshow(image)
    else:
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    if dimu2 is not None:
        ax.plot(
            x,
            dimu2,
            "r-",
            alpha=1.0 if win <= 0 else 0.15,
            label=r"$\text{dim}(u_{2})$" if win <= 0 else None,
        )
        if win > 0:
            new_x, new_dim = ema(x, dimu2, win=win)
            ax.plot(
                new_x,
                new_dim,
                "r-",
                alpha=1.0,
                label=r"$\text{dim}(u_{2})$",
            )
    if dimu4 is not None:
        ax.plot(
            x,
            dimu4,
            "g--",
            alpha=1.0 if win <= 0 else 0.15,
            label=r"$\text{dim}(u_{4})$" if win <= 0 else None,
        )
        if win > 0:
            new_x, new_dim = ema(x, dimu4, win=win)
            ax.plot(
                new_x,
                new_dim,
                "g--",
                alpha=1.0,
                label=r"$\text{dim}(u_{4})$",
            )
    if dimu6 is not None:
        ax.plot(
            x,
            dimu6,
            "b-.",
            alpha=1.0 if win <= 0 else 0.15,
            label=r"$\text{dim}(u_{6})$" if win <= 0 else None,
        )
        if win > 0:
            new_x, new_dim = ema(x, dimu6, win=win)
            ax.plot(
                new_x,
                new_dim,
                "b-.",
                alpha=1.0,
                label=r"$\text{dim}(u_{6})$",
            )

    ax.set(xlabel=name, ylabel="canonical dimensions")
    ax.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    if suffix is None:
        plt.savefig(output_dir / "canonical_dimensions_snr.pdf")
    else:
        plt.savefig(output_dir / f"canonical_dimensions_{suffix}.pdf")


def plot_ratio_scan(
    groups: pd.DataFrame,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the canonical dimensions as a function of a particular quantity of interest.

    Parameters
    ----------
    groups : pd.DataFrame
        The grouped dataframe containing the points to plot.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    _, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    ax.fill_between(
        groups.index,
        groups[("dimu2", "mean")] - groups[("dimu2", "std")],
        groups[("dimu2", "mean")] + groups[("dimu2", "std")],
        color="r",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups[("dimu2", "mean")],
        "r-",
        label=r"$\text{dim}(u_{2})$",
    )
    ax.fill_between(
        groups.index,
        groups[("dimu4", "mean")] - groups[("dimu4", "std")],
        groups[("dimu4", "mean")] + groups[("dimu4", "std")],
        color="g",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups[("dimu4", "mean")],
        "g--",
        label=r"$\text{dim}(u_{4})$",
    )
    ax.fill_between(
        groups.index,
        groups[("dimu6", "mean")] - groups[("dimu6", "std")],
        groups[("dimu6", "mean")] + groups[("dimu6", "std")],
        color="b",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups[("dimu6", "mean")],
        "b-.",
        label=r"$\text{dim}(u_{6})$",
    )

    ax.set(xlabel="ratio ($q = p / n$)", ylabel="canonical dimensions")
    ax.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    if suffix is None:
        plt.savefig(output_dir / "canonical_dimensions_ratio.pdf")
    else:
        plt.savefig(output_dir / f"canonical_dimensions_ratio_{suffix}.pdf")


def plot_localization(
    data: dict[str, Any],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> tuple[float, float, float, float, float]:
    """
    Plot the localization of the components of the eigenvectors in the UV and IR.

    Parameters
    ----------
    data : dist[str, Any]
        The eigenvalues and eigenvectors.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.

    Returns
    -------
    tuple[float, float, float, float, float]
        The values of:

        - the ratio of the histograms at the average value of the UV (0.0),
        - the average value of the UV distribution,
        - the standard deviation of the UV distribution.
        - the average value of the IR distribution,
        - the standard deviation of the IR distribution.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find the corresponding index of the eigenvalues/vectors
    idx = int(np.argmin(np.abs(np.array(data["evl"]) - data["lplus_mp"])))

    # Consider 100 eigenvectors in the UV and the IR
    n_right = len(data["evl"]) - idx
    n_left = 100 - n_right

    evc_uv = np.array(data["evc"]).T[:100].ravel()
    evc_ir = np.array(data["evc"]).T[idx - n_left : idx + n_right].ravel()

    # Plot the results
    fig = plt.figure(figsize=(14, 5))
    axs = fig.subplot_mosaic(
        [["hist", "joint"], ["ratio", "joint"]],
        height_ratios=[2, 1],
        sharex=True,
    )

    axs["hist"].axes.get_xaxis().set_visible(False)
    axs["hist"].axhline(0.0, color="k", ls="dotted", alpha=0.15)
    axs["hist"].axvline(0.0, color="k", ls="dotted", alpha=0.15)

    uv_bins, uv_edges, _ = axs["hist"].hist(
        evc_uv,
        bins=2 * int(np.sqrt(len(data["evl"]))),
        color="k",
        label="UV",
        density=True,
        histtype="step",
    )
    axs["hist"].axvline(evc_uv.mean(), color="k", ls="dashed")

    ir_bins, _, _ = axs["hist"].hist(
        evc_ir,
        bins=uv_edges,
        color="r",
        label="IR",
        density=True,
        histtype="step",
    )
    axs["hist"].axvline(evc_ir.mean(), color="r", ls="dashed")
    axs["hist"].legend(loc="best", ncol=1, frameon=False)
    axs["hist"].set_ylabel("components")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ir_bins / uv_bins

        axs["ratio"].plot(uv_edges[:-1], ratio, color="k")
        axs["ratio"].axhline(1.0, color="k", ls="dashed", alpha=0.15)
        axs["ratio"].axvline(0.0, color="k", ls="dashed", alpha=0.15)
        axs["ratio"].set_ylim(0.8, 1.2)
        axs["ratio"].set_yticks([0.9, 1.0, 1.1])

        axs["ratio"].set_ylabel("IR / UV")

    axs["joint"].hist2d(
        evc_ir, evc_uv, bins=uv_edges, density=True, cmap="turbo"
    )
    axs["joint"].yaxis.tick_right()
    axs["joint"].yaxis.set_label_position("right")
    axs["joint"].set(xlabel="IR", ylabel="UV")

    plt.subplots_adjust(hspace=0, wspace=0)
    if suffix is None:
        plt.savefig(output_dir / "localization_plot.pdf", dpi=300)
    else:
        plt.savefig(output_dir / f"localization_plot_{suffix}.pdf", dpi=300)

    # Find the point around the average value of the UV distribution
    idx = np.argmin(np.abs(uv_edges - evc_uv.mean()))

    return (
        float(ratio[idx]),
        float(evc_uv.mean()),
        float(evc_uv.std()),
        float(evc_ir.mean()),
        float(evc_ir.std()),
    )


def plot_eigenvalues(
    data: dict[str, Any],
    suffix: str | None = None,
    zoom: bool = False,
    output_dir: str | Path = "plots",
):
    """
    Plot the eigenvalues of the distribution.

    Parameters
    ----------
    data : dist[str, Any]
        The eigenvalues and eigenvectors.
    suffix : str
        The name to postpone to the file name.
    zoom : bool
        Add an inset axis to zoom on the IR region. By default `False`.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    evls = np.array(data["evl"])
    ax.hist(
        evls,
        bins=2 * int(np.sqrt(len(evls))),
        color="k",
        density=True,
        histtype="step",
    )

    ax.axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax.axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\mu$")

    if zoom:
        max_evls = max(evls)
        ax_inset = ax.inset_axes([1.5, 0.85, 1.5, 1.5], transform=ax.transData)
        ax_inset.hist(
            evls,
            bins=2 * int(np.sqrt(len(evls))),
            color="k",
            density=True,
            histtype="step",
        )
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(0.85 * max_evls, 1.02 * max_evls)
        ax_inset.set_ylim(-0.01, 0.1)
        ax_inset.set_title("IR")
        ax_inset.tick_params(labelsize=12)
        ax.indicate_inset_zoom(ax_inset, edgecolor="k")

    if suffix is None:
        plt.savefig(output_dir / "eigenvalues_dist.pdf")
    else:
        plt.savefig(output_dir / f"eigenvalues_dist_{suffix}.pdf")


def plot_localization_scan(
    snrs: ArrayLike,
    ratios: ArrayLike,
    uv_stds: ArrayLike,
    ir_means: ArrayLike,
    ir_stds: ArrayLike,
    output_dir: str | Path = "plots",
):
    """
    Plot values of the localization of the eigenvector components as a functions of the signal-to-noise ratio.

    Parameters
    ----------
    snrs : ArrayLike
        The list of the signal-to-noise ratio values.
    ratios : ArrayLike
        The list of ratios at the average of the UV distributions.
    uv_stds : ArrayLike
        The standard deviations of the UV distribution.
    ir_means : ArrayLike
        The mean values of the IR distribution.
    ir_stds : ArrayLike
        The standard deviations of the IR distribution.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots(
        ncols=2, nrows=2, figsize=(7 * 2, 5 * 2), layout="constrained"
    )
    ax = ax.ravel()

    ax[0].plot(snrs, ratios, "kx")
    ax[0].plot(snrs, ratios, "k--", alpha=0.5)
    ax[0].set(xlabel="signal-to-noise ratio ($\\beta$)", ylabel="ratio")
    ax[0].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax[0].axhline(1.0, ls="dashed", color="k", alpha=0.15)
    ax[1].plot(snrs, ir_means, "rx")
    ax[1].plot(snrs, ir_means, "r--", alpha=0.5)
    ax[1].set(xlabel="signal-to-noise ratio ($\\beta$)", ylabel="IR (average)")
    ax[1].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax[2].plot(snrs, ir_stds, "rx")
    ax[2].plot(snrs, ir_stds, "r--", alpha=0.5)
    ax[2].set(
        xlabel="signal-to-noise ratio ($\\beta$)",
        ylabel="IR (standard deviation)",
    )
    ax[2].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax[3].plot(snrs, np.array(ir_stds) / np.array(uv_stds), "kx")
    ax[3].plot(snrs, np.array(ir_stds) / np.array(uv_stds), "k--", alpha=0.5)
    ax[3].set(
        xlabel="signal-to-noise ratio ($\\beta$)",
        ylabel="IR / UV (standard deviation)",
    )
    ax[3].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    ax[3].axhline(1.0, ls="dashed", color="k", alpha=0.15)

    plt.savefig(output_dir / "localization_scan.pdf")


def plot_trajectories(
    data: dict[str, Any],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the FRG trajectories.

    Parameters
    ----------
    data : dict[str, Any]
        The list of the signal-to-noise ratio values.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    axs = fig.subplot_mosaic(
        [["evo", "u2u4"], ["evo", "u4u6"], ["evo", "u2u6"]],
        height_ratios=[1, 1, 1],
    )

    axs["evo"].plot(data["k2"], data["u2"], "r-", label="$u_2$")
    axs["evo"].plot(data["k2"], data["u4"], "g--", label="$u_4$")
    axs["evo"].plot(data["k2"], data["u6"], "b-.", label="$u_6$")

    axs["evo"].set(xlabel="$k^2$", ylabel="couplings")

    axs["evo"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["evo"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["evo"].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )
    axs["evo"].legend(loc="best", ncol=1, frameon=False)

    # Show separate couplings
    ax_inset = axs["evo"].inset_axes([0.2, 0.75, 0.5, 0.2])
    ax_inset.plot(data["k2"], data["u2"], "r-")
    ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.set_title("$u_2$")
    ax_inset.tick_params(labelsize=12)
    ax_inset.yaxis.get_offset_text().set_fontsize(12)
    ax_inset.xaxis.get_offset_text().set_fontsize(12)
    ax_inset.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    ax_inset = axs["evo"].inset_axes([0.3, 0.45, 0.5, 0.2])
    ax_inset.plot(data["k2"], data["u4"], "g--")
    ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.set_title("$u_4$")
    ax_inset.tick_params(labelsize=12)
    ax_inset.yaxis.get_offset_text().set_fontsize(12)
    ax_inset.xaxis.get_offset_text().set_fontsize(12)
    ax_inset.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    ax_inset = axs["evo"].inset_axes([0.4, 0.15, 0.5, 0.2])
    ax_inset.plot(data["k2"], data["u6"], "b-.")
    ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax_inset.set_title("$u_6$")
    ax_inset.tick_params(labelsize=12)
    ax_inset.yaxis.get_offset_text().set_fontsize(12)
    ax_inset.xaxis.get_offset_text().set_fontsize(12)
    ax_inset.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    # Show trakectories
    axs["u2u4"].plot(data["u2"], data["u4"], "k-")
    axs["u2u4"].plot([data["u2"][0]], [data["u4"][0]], "bo", label="UV")
    axs["u2u4"].plot([data["u2"][-1]], [data["u4"][-1]], "ro", label="IR")
    axs["u2u4"].set(xlabel="$u_2$", ylabel="$u_4$")
    axs["u2u4"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u4"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u4"].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    axs["u4u6"].plot(data["u4"], data["u6"], "k-")
    axs["u4u6"].plot([data["u4"][0]], [data["u6"][0]], "bo", label="UV")
    axs["u4u6"].plot([data["u4"][-1]], [data["u6"][-1]], "ro", label="IR")
    axs["u4u6"].set(xlabel="$u_4$", ylabel="$u_6$")
    axs["u4u6"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u4u6"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u4u6"].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )
    axs["u4u6"].legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), ncols=1, frameon=False
    )

    axs["u2u6"].plot(data["u2"], data["u6"], "k-")
    axs["u2u6"].plot([data["u2"][0]], [data["u6"][0]], "bo", label="UV")
    axs["u2u6"].plot([data["u2"][-1]], [data["u6"][-1]], "ro", label="IR")
    axs["u2u6"].set(xlabel="$u_2$", ylabel="$u_6$")
    axs["u2u6"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u6"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u6"].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    if suffix is None:
        plt.savefig(output_dir / "frg_equations.pdf")
    else:
        plt.savefig(output_dir / f"frg_equations_{suffix}.pdf")


def plot_symmetry_surface(
    phases_ir: list[int],
    u2: ArrayLike,
    u4: ArrayLike,
    u6: ArrayLike,
    phases_uv: list[int] | None = None,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the symmetry surface of the FRG equations.

    Parameters
    ----------
    phases_ir : list[int]
        The classification of the phase in the IR region (1 is symmetric, 0 is broken symmetry).
    u2 : ArrayLike
        The list of values of the quadratic coupling.
    u4 : ArrayLike
        The list of values of the quartic coupling.
    u6 : ArrayLike
        The list of values of the sextic coupling.
    phases_uv : list[int], optional
        The classification of the phase in the UV region (1 is symmetric, 0 is broken symmetry).
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Select only symmetric points in the UV
    if phases_uv is not None:
        phases_ir = np.array(phases_ir)[np.array(phases_uv).astype("bool")]
        u2 = np.array(u2)[np.array(phases_uv).astype("bool")]
        u4 = np.array(u4)[np.array(phases_uv).astype("bool")]
        u6 = np.array(u6)[np.array(phases_uv).astype("bool")]

    # Plot the points
    _, ax = plt.subplots(ncols=3, figsize=(21, 5), layout="constrained")

    col = [
        (1.0, 0.0, 0.0, 1.0) if p > 0 else (0.0, 0.0, 0.0, 0.15)
        for p in phases_ir
    ]
    ax[0].scatter(u2, u4, c=col)
    ax[0].set(xlabel="$u_2$", ylabel="$u_4$")
    ax[0].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[0].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[0].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    ax[1].scatter(u4, u6, c=col)
    ax[1].set(xlabel="$u_4$", ylabel="$u_6$")
    ax[1].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[1].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[1].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=(0.0, 0.0, 0.0, 0.15),
            marker="o",
            lw=0,
            label="broken symmetry",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=(1.0, 0.0, 0.0, 1.0),
            marker="o",
            lw=0,
            label="symmetric phase",
        ),
    ]
    ax[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        ncols=2,
    )

    ax[2].scatter(u2, u6, c=col)
    ax[2].set(xlabel="$u_2$", ylabel="$u_6$")
    ax[2].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[2].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[2].ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )

    if suffix is None:
        plt.savefig(output_dir / "frg_symmetry_surface.pdf")
    else:
        plt.savefig(output_dir / f"frg_symmetry_surface_{suffix}.pdf")


def plot_symmetry_size(
    sizes: dict[str, float],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the relative size of the symmetric phase.

    Parameters
    ----------
    sizes: dict[str, float]
        A collection containing the SNR as key and the relative size of the symmetric region as values.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    snr = np.array(list(sizes.keys()))
    val = 100 * np.array(list(sizes.values()))
    ax.plot(snr, val, "kx")
    ax.plot(snr, val, "k--", alpha=0.15)
    ax.set(xlabel="signal-to-noise ratio ($\\beta$)", ylabel="relative size")
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.PercentFormatter(decimals=1, is_latex=True)
    )

    if suffix is None:
        plt.savefig(output_dir / "frg_symmetry_size.pdf")
    else:
        plt.savefig(output_dir / f"frg_symmetry_size_{suffix}.pdf")


def plot_potential(
    x: ArrayLike,
    u2: dict[float, ArrayLike],
    u4: dict[float, ArrayLike],
    n: int,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Plot the relative size of the symmetric phase.

    Parameters
    ----------
    x : ArrayLike
        The list of point to be evaluated.
    u2 : dict[float, ArrayLike]
        A collection of quadratic couplings per SNR.
    u4 : dict[float, ArrayLike]
        A collection of quartic couplings per SNR.
    n : int
        The index of the sample to consider.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(nrows=2, figsize=(9, 10), layout="constrained")

    ax[0].axhline(0.0, color="k", ls="dashed", alpha=0.15)
    ax[0].axvline(0.0, color="k", ls="dashed", alpha=0.15)
    ax[0].ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )

    # Define the potential
    def _potential(
        x: np.ndarray,
        u2: dict[float, list[float]],
        u4: dict[float, list[float]],
        snr: float,
        n: int = 0,
    ) -> np.ndarray:
        y = u2[snr][n] * x**2 + u4[snr][n] * x**4
        return y - y.min()

    # Plot the curves with different styles
    colors = mpl.colormaps["tab10"]
    lines = [
        (0, (1, 8)),
        (0, (1, 1)),
        (0, (5, 8)),
        (0, (5, 1)),
        (5, (10, 3)),
        (0, (3, 10, 1, 10)),
        "dashed",
        "dotted",
        "dashdot",
        "solid",
    ]
    lines = [
        (0, (1, 1)),
        (0, (1, 5)),
        (0, (5, 1)),
        (0, (5, 5)),
        (0, (1, 3, 5, 3)),
        (0, (5, 1, 3, 1, 5)),
        (0, (2, 1, 3, 1, 2)),
        (0, (2, 2, 5, 2, 2)),
        "dashdot",
        "solid",
    ]
    y_max = -np.inf
    y_collection = np.zeros((len(u2.keys()), len(x)))
    for m, snr in enumerate(u2.keys()):
        y = _potential(x, u2=u2, u4=u4, snr=snr, n=n)
        y_collection[m] = y
        y_max = np.maximum(y_max, y[np.argmin(np.abs(x))])
        ax[0].plot(
            x,
            y,
            ls=lines[m],
            color=colors(snr * 2.0),
            label=f"SNR = {snr:.2f}",
        )

    ax[0].set(xlabel="$M$", ylabel="$U$")
    ax[0].set_ylim(-1.0e-7, 1.15 * y_max)
    ax[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    # Plot the surface
    img = ax[1].imshow(
        y_collection.clip(0.0, 1.15 * y_max),
        aspect=50 * (7 / 5),
        cmap="viridis",
    )
    ax[1].set(xlabel="$M$", ylabel="signal-to-noise ratio ($\\beta$)")
    ax[1].set_yticks(range(len(u2.keys())))
    ax[1].set_yticklabels([f"{x:.2f}" for x in u2.keys()])
    l_idx = np.argmin(np.abs(x + 1.0))
    r_idx = np.argmin(np.abs(x - 1.0))
    ax[1].set_xticks([l_idx, 500, r_idx])
    ax[1].set_xticklabels([f"{x:.2f}" for x in x[ax[1].get_xticks()]])
    format = mpl.ticker.ScalarFormatter(useMathText=True)
    format.set_scientific(True)
    format.set_powerlimits((0, 0))
    cbar = fig.colorbar(img, ax=ax[1], shrink=0.9, pad=-0.35, format=format)
    cbar.set_label("$U$", labelpad=10)

    if suffix is None:
        plt.savefig(output_dir / "frg_potential.pdf")
    else:
        plt.savefig(output_dir / f"frg_potential_{suffix}.pdf")


def direct_relative_adherence(
    data: dict[str, Any],
    thresh: float = 0.5,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
):
    """
    Compute the direct relative adherence.

    Parameters
    ----------
    data : dict[str, Any]
        The results of the computation of the canonical dimensions.
    thresh : float
        The value of the threshold on the distribution to be considered "bulk". By default `0.5`.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Compute the point between the max and the start
    k2 = np.array(data["k2"])
    idx, _, _ = compute_roi(data=data, thresh=thresh)
    _, dimu4_interp, _ = interp_canonical_dimensions(data, idx)
    dimu4_emp = dimu4_interp(k2)

    # Compute the proxy
    ratios = np.linspace(0.10, 0.99, num=10)
    proxy = MarchenkoPastur(ratio=0.9, sigma=1.0)
    best_distance = np.inf
    best_dimu4 = np.zeros_like(k2)
    for ratio in ratios:
        sigma = np.sqrt(1 / (4.0 * np.sqrt(ratio)) / data["m2"])
        mp = MarchenkoPastur(ratio=ratio, sigma=sigma)

        # Canonical dimensions
        _, dimu4, _, _ = mp.canonical_dimensions(k2).T
        distance = np.abs(dimu4 - dimu4_emp).max()
        if distance < best_distance:
            best_distance = distance
            best_dimu4 = deepcopy(dimu4)
            proxy = deepcopy(mp)

    # Compute the adherence
    adherence = np.zeros_like(k2)
    for n in range(1, len(k2)):
        adherence[n] = (best_dimu4[:n] - dimu4_emp[:n]).min()

    # Plot
    _, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    ax.plot(k2[1:], adherence[1:], "r-", label="local inverse adherence")
    ax.set(xlabel="$k^2$", ylabel="$\\zeta^{-1}_{k^2}$")

    ax_twin = ax.twinx()

    slope_change = np.abs(np.diff(adherence))
    idx_slope = slope_change.nonzero()[0][-1]
    ax.axvline(data["k2"][idx_slope], color="r", ls="dashed")
    pos = (adherence.max() - adherence.min()) / 2.0 + adherence.min()
    ax.text(
        data["k2"][idx_slope] * 0.99,
        ha="right",
        y=pos,
        s=f"$k^2_c$ = {data['k2'][idx_slope]:.2f}",
        color="r",
        rotation=90,
    )

    ax_twin.plot(k2, np.array(data["dist"]), "k-")
    full_k2 = np.linspace(0.0, k2[-1], num=1000)
    ax_twin.plot(full_k2, proxy.ipdf(full_k2), "k--", alpha=0.75)
    ax_twin.set_ylabel("PDF")

    # Legend
    custom_lines = [
        mpl.lines.Line2D([0], [0], color="r"),
        mpl.lines.Line2D([0], [0], color="k", ls="dashed", alpha=0.5),
        mpl.lines.Line2D([0], [0], color="k"),
    ]

    ax.legend(
        handles=custom_lines,
        labels=[
            "$\\zeta^{-1}_{k^2}$",
            "proxy",
            "data",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        ncols=3,
    )

    if suffix is None:
        plt.savefig(output_dir / "adherence.pdf")
    else:
        plt.savefig(output_dir / f"adherence_{suffix}.pdf")

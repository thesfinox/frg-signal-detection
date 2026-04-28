"""Functions used for data analysis."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from frg.distributions.distributions import Distribution, MarchenkoPastur

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any

    from jaxtyping import Float, Integer
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    # Dummy variables for jaxtyping to prevent Ruff F821 errors
    n: int = 1
    nV: int = 1
    w: int = 1
    p: int = 1
    SNR: int = 1


def _ema(
    x: Float[np.ndarray, n],
    y: Float[np.ndarray, n],
    win: int,
) -> tuple[Float[np.ndarray, nV], Float[np.ndarray, nV]]:
    """Exponential movin average (EMA).

    Parameters
    ----------
    x : array of floats
        The list of values (shape: :math:`n`).
    y : array of floats
        The list of values (shape: :math:`n`).
    win : int
        The scaling width.

    Returns
    -------
    tuple of arrays of floats
        The smoothed list of x and y (shape: ).
    """
    x_arr: Float[np.ndarray, n] = np.array(x)
    y_arr: Float[np.ndarray, n] = np.array(y)

    # Window
    win_list: Float[np.ndarray, w] = np.ones((win,)) / win

    # Convolution
    new_x: Float[np.ndarray, nV] = np.convolve(x_arr, win_list, mode="valid")
    new_y: Float[np.ndarray, nV] = np.convolve(y_arr, win_list, mode="valid")
    return new_x, new_y


def compute_roi(
    data: dict[str, Any],
    thresh: float = 0.5,
    analytic: bool = False,
) -> tuple[int, int, int]:
    """Compute the indices of the region of interest, and its initial and final points.

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
    top: int = int(np.argmax(data["dist"]))
    if top == 0:
        return 0, 0, 0
    if analytic:
        return 1, 0, top
    start: int = int(np.argmin(np.abs(np.array(data["dist"][:top]) - thresh)))

    idx: int = start + (top - start) // 2

    return int(idx), int(start), int(top)


def interp_canonical_dimensions(
    data: dict[str, Any],
    idx: int,
) -> tuple[Callable, Callable, Callable]:
    """Interpolate the behaviour of the canonical dimensions.

    Parameters
    ----------
    data : dict[str, Any]
        The experimental data.
    idx : int
        The index of the starting point.

    Returns
    -------
    tuple[Callable, Callable, Callable]
        The interpolated canonical dimensions.
    """
    stop: int = int(np.argmin(np.abs(np.array(data["k2"]) - 0.7)))
    dimu2_interp: Callable = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu2"][idx:stop], 1),
    )
    dimu4_interp: Callable = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu4"][idx:stop], 1),
    )
    dimu6_interp: Callable = np.poly1d(
        np.polyfit(data["k2"][idx:stop], data["dimu6"][idx:stop], 1),
    )
    return dimu2_interp, dimu4_interp, dimu6_interp


def extract_interp_values(
    data: dict[str, Any],
    thresh: float = 0.5,
    deep_ir: bool = False,
) -> tuple[float, float, float, float]:
    r"""Extract the interpolated values.

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
        The values of :math:`k^2`, :math:`\text{dim}(u_{2})`, :math:`\text{dim}(u_{4})`, and :math:`\text{dim}(u_{6})` at the reference scale.
    """
    idx: int
    idx, _, _ = compute_roi(data, thresh, analytic=deep_ir)
    dimu2_interp: Callable
    dimu4_interp: Callable
    dimu6_interp: Callable
    dimu2_interp, dimu4_interp, dimu6_interp = interp_canonical_dimensions(
        data,
        idx,
    )

    k2: float = float(data["k2"][idx]) if not deep_ir else 0.0
    return (
        k2,
        float(dimu2_interp(k2)),
        float(dimu4_interp(k2)),
        float(dimu6_interp(k2)),
    )


def canonical_dimensions_argsort(
    x: Float[np.ndarray, n] | list[float],
    dimu2: Float[np.ndarray, n] | list[float],
    dimu4: Float[np.ndarray, n] | list[float],
    dimu6: Float[np.ndarray, n] | list[float],
) -> tuple[
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
]:
    """Index sort the signal to noise ratio and the canonical dimensions.

    Parameters
    ----------
    x : array of floats
        The quantity of interest (shape: :math:`n`).
    dimu2 : array of floats
        The canonical dimension of the quadratic coupling (shape: :math:`n`).
    dimu4 : array of floats
        The canonical dimension of the quartic coupling (shape: :math:`n`).
    dimu6 : array of floats
        The canonical dimension of the sextic coupling (shape: :math:`n`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing the signal-to-noise ratio and the canonical dimensions in the same order as the input parameters.
    """
    x_arr: Float[np.ndarray, n] = np.array(x)
    dimu2_arr: Float[np.ndarray, n] = np.array(dimu2)
    dimu4_arr: Float[np.ndarray, n] = np.array(dimu4)
    dimu6_arr: Float[np.ndarray, n] = np.array(dimu6)

    idx: Integer[np.ndarray, n] = np.argsort(x_arr)

    return x_arr[idx], dimu2_arr[idx], dimu4_arr[idx], dimu6_arr[idx]


def canonical_dimensions_files(
    path: str | Path,
    glob: str = "*.json",
    analytic: bool = False,
) -> tuple[
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
]:
    """Open multiple files and stores the canonical dimensions.

    Parameters
    ----------
    path : str | Path
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
    x: list[float] = []
    dimu2: list[float] = []
    dimu4: list[float] = []
    dimu6: list[float] = []
    scale: list[float] = []

    files: Generator[Path, None, None] = Path(path).glob(glob)

    for file in files:
        with Path(file).open() as f:
            data: dict[str, Any] = json.load(f)
        add_values(
            extract_interp_values(data, deep_ir=analytic),
            scale,
            dimu2,
            dimu4,
            dimu6,
        )

        value: re.Match[str] | None = re.search(r"[0-9][.][0-9]*", file.stem)
        if value:
            x.append(float(value.group()))

    return canonical_dimensions_argsort(x, dimu2, dimu4, dimu6)


def canonical_dimensions_files_poiss(
    path: str | Path,
    glob: str = "*.json",
) -> tuple[
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
    Float[np.ndarray, n],
]:
    """Open multiple files and stores the canonical dimensions in case of Poisson noise.

    Parameters
    ----------
    path : str | Path
        The path to the directory containing the files.
    glob : str
        The global pattern to open. By default `"*.json"`.

    Returns
    -------
    tuple[np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing the Gaussian signal-to-noise ratio, the Poisson parameter, and the canonical dimensions.
    """
    snr: list[float] = []
    lam: list[float] = []
    dimu2: list[float] = []
    dimu4: list[float] = []
    dimu6: list[float] = []
    scale: list[float] = []

    files: Generator[Path, None, None] = Path(path).glob(glob)

    for file in files:
        with Path(file).open() as f:
            data: dict[str, Any] = json.load(f)
        add_values(
            extract_interp_values(data),
            scale,
            dimu2,
            dimu4,
            dimu6,
        )

        snr_value: re.Match[str] | None = re.search(
            r"snr=[0-9]*[.][0-9]*",
            file.stem,
        )
        if snr_value:
            snr_float: float = float(snr_value.group()[4:])
            snr.append(snr_float)
        lam_value: re.Match[str] | None = re.search(
            r"lam=[0-9]*[.][0-9]*",
            file.stem,
        )
        if lam_value:
            lam_float: float = float(lam_value.group()[4:])
            lam.append(lam_float)

    return (
        np.array(snr),
        np.array(lam),
        np.array(dimu2),
        np.array(dimu4),
        np.array(dimu6),
    )


def canonical_dimensions_ratio_files(
    path: str | Path,
    glob: str = "*.json",
    analytic: bool = False,
) -> pd.DataFrame:
    """Open multiple files as a function of ratio and seed and stores the canonical dimensions.

    Parameters
    ----------
    path : str | Path
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
    ratio_l: list[float] = []
    seed_l: list[int] = []
    dimu2: list[float] = []
    dimu4: list[float] = []
    dimu6: list[float] = []
    scale: list[float] = []

    files: Generator[Path, None, None] = Path(path).glob(glob)

    for file in files:
        with Path(file).open() as f:
            data: dict[str, Any] = json.load(f)
        add_values(
            extract_interp_values(data, deep_ir=analytic),
            scale,
            dimu2,
            dimu4,
            dimu6,
        )

        ratio_match: re.Match[str] | None = re.search(
            r"_ratio=[0-9][.][0-9]*?_",
            str(file),
        )
        if ratio_match:
            ratio_str: str = ratio_match.group()[1:-1]
            ratio: float = float(ratio_str.rsplit("=", maxsplit=1)[-1])
            ratio_l.append(ratio)

        seed_match: re.Match[str] | None = re.search(r"_seed=[0-9]*", str(file))
        if seed_match:
            seed_str: str = seed_match.group()[1:]
            seed: int = int(seed_str.rsplit("=", maxsplit=1)[-1])
            seed_l.append(seed)

    return pd.DataFrame({
        "ratio": ratio_l,
        "seed": seed_l,
        "dimu2": dimu2,
        "dimu4": dimu4,
        "dimu6": dimu6,
    })


def add_values(
    interp_values: tuple[float, float, float, float],
    scale: list[float],
    dimu2: list[float],
    dimu4: list[float],
    dimu6: list[float],
) -> None:
    r"""Add values to lists.

    Parameters
    ----------
    interp_values : tuple[float, float, float, float]
        The interpolated values of :math:`k^2`, :math:`\text{dim}(u_{2})`, :math:`\text{dim}(u_{4})`, :math:`\text{dim}(u_{6})`.
    scale : list[float]
        The list of values of the reference scale :math:`k^2`.
    dimu2 : list[float]
        The list of values of :math:`\text{dim}(u_{2})`.
    dimu4 : list[float]
        The list of values of :math:`\text{dim}(u_{4})`.
    dimu6 : list[float]
        The list of values of :math:`\text{dim}(u_{6})`.
    """
    k2: float
    dimu2_value: float
    dimu4_value: float
    dimu6_value: float
    k2, dimu2_value, dimu4_value, dimu6_value = interp_values
    scale.append(k2)
    dimu2.append(dimu2_value)
    dimu4.append(dimu4_value)
    dimu6.append(dimu6_value)


def plot_distribution(
    dist: Distribution,
    output_dir: str | Path = "plots",
) -> None:
    """Plot distributions.

    Parameters
    ----------
    dist : Distribution
        The distribution to show.
    output_dir : str | Path
        The output directory of the plots. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig: Figure
    ax: np.ndarray
    fig, ax = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")

    # Show axes
    ax[0].axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[0].axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$\mu$")
    ax[1].axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[1].axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax[1].set(xlabel="$k^2$", ylabel="$\rho$")

    if isinstance(dist, MarchenkoPastur):
        # PDF
        x: Float[np.ndarray, 5000] = np.linspace(
            0.0,
            1.05 * dist.lplus,
            num=5000,
        )
        y: Float[np.ndarray, 5000] = dist.pdf(x)
        ax[0].plot(x, y, "k-")

        ax_inset = ax[0].inset_axes(
            [0.35, 1.55, 1.5, 1.5],
            transform=ax[0].transData,
        )
        ax_inset.plot(x, y, "k-")
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(0.1 * dist.lminus, 20 * dist.lminus)
        ax_inset.set_title("UV")
        ax_inset.tick_params(labelsize=12)
        ax[0].indicate_inset_zoom(ax_inset, edgecolor="k")

        ax_inset = ax[0].inset_axes(
            [2.5, 0.75, 1.5, 1.5],
            transform=ax[0].transData,
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
        x_inv: Float[np.ndarray, 1000] = np.linspace(0.0, 3.0, num=1000)
        y_inv: Float[np.ndarray, 1000] = dist.ipdf(x_inv)
        ax[1].plot(x_inv, y_inv, "k-")

        ax_inset = ax[1].inset_axes(
            [0.95, 0.45, 1.15, 0.35],
            transform=ax[1].transData,
        )
        ax_inset.plot(x_inv, y_inv, "k-")
        ax_inset.axhline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.axvline(0.0, ls="dashed", color="k", alpha=0.15)
        ax_inset.set_xlim(-0.02, 0.35)
        ax_inset.set_title("IR")
        ax_inset.tick_params(labelsize=12)
        ax[1].indicate_inset_zoom(ax_inset, edgecolor="k")

        plt.savefig(
            out_dir
            / f"marchenkopastur_ratio={dist.ratio}_sigma={dist.var}.pdf",
        )
    else:
        evls: Float[np.ndarray, p] = dist.eigenvalues_

        # PDF
        ax[0].hist(
            evls,
            bins=2 * int(np.sqrt(len(evls))),
            color="b",
            alpha=0.5,
            density=True,
        )

        x_emp: Float[np.ndarray, 1000] = np.linspace(
            0.0,
            1.05 * dist.lplus,
            num=1000,
        )
        y_emp: Float[np.ndarray, 1000] = dist.pdf(x_emp)
        ax[0].plot(x_emp, y_emp, "k-")

        # PDF of the inverse
        x_inv_emp: Float[np.ndarray, 1000] = np.linspace(0.0, 3.0, num=1000)
        y_inv_emp: Float[np.ndarray, 1000] = dist.ipdf(x_inv_emp)
        ax[1].plot(x_inv_emp, y_inv_emp, "k-")

        plt.savefig(
            out_dir
            / f"empirical_dist_ratio={dist.ratio}_sigma={dist.var}_nsamples={dist.n_samples}.pdf",
        )


def plot_canonical_dimensions(
    data: dict[str, Any],
    thresh: float = 0.5,
    suffix: str | None = None,
    analytic: bool = False,
    output_dir: str | Path = "plots",
    ratio: float | None = None,
    var: float | None = None,
) -> None:
    r"""Plot a single instance of the canonical dimensions.

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
    ratio : float, optional
        Marchenko-Pastur ratio :math:`q = p/n`. If provided together with ``var``,
        displayed as the legend title.
    var : float, optional
        Marchenko-Pastur variance :math:`\\sigma^2`. If provided together with
        ``ratio``, displayed as the legend title.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    # Compute the point between the max and the start
    idx: int
    start: int
    top: int
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
        dimu2_interp: Callable
        dimu4_interp: Callable
        dimu6_interp: Callable
        dimu2_interp, dimu4_interp, dimu6_interp = interp_canonical_dimensions(
            data,
            idx,
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

        k2_idx: float = data["k2"][idx]
        ax.plot([k2_idx], [dimu2_interp(k2_idx)], "ro")
        ax.text(
            x=k2_idx * 1.1,
            y=dimu2_interp(k2_idx),
            s=rf"$\text{{dim}}(u_2) = {dimu2_interp(k2_idx):.2f}$",
            color="r",
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.plot([k2_idx], [dimu4_interp(k2_idx)], "go")
        ax.text(
            x=k2_idx * 1.1,
            y=dimu4_interp(k2_idx),
            s=rf"$\text{{dim}}(u_4) = {dimu4_interp(k2_idx):.2f}$",
            color="g",
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.plot([k2_idx], [dimu6_interp(k2_idx)], "bo")
        ax.text(
            x=k2_idx * 1.1,
            y=dimu6_interp(k2_idx),
            s=rf"$\text{{dim}}(u_6) = {dimu6_interp(k2_idx):.2f}$",
            color="b",
            fontsize=10,
            ha="left",
            va="top",
        )

    ax.set_xlabel(r"$k^2$")
    ax.set_ylabel("canonical dimensions")
    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    _legend_title: str | None = (
        rf"$q = {ratio},\ \sigma^2 = {var}$"
        if ratio is not None and var is not None
        else None
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        title=_legend_title,
    )
    ax2 = ax.twinx()
    ax2.plot(data["k2"], data["dist"], "k--")
    ax2.set(ylabel=r"$\rho$")

    if suffix is None:
        plt.savefig(out_dir / "canonical_dimensions.pdf")
    else:
        plt.savefig(out_dir / f"canonical_dimensions_{suffix}.pdf")


def plot_canonical_dimensions_eigenvalues(
    data: dict[str, Any],
    dist: Distribution,
    thresh: float = 0.5,
    suffix: str | None = None,
    analytic: bool = False,
    output_dir: str | Path = "plots",
) -> None:
    r"""Plot a single instance of the canonical dimensions on the eigenvalue spectrum.

    The momentum scale :math:`k^2` is mapped back to the eigenvalue axis
    :math:`\lambda` via the inverse of the change of variables used in
    :meth:`~frg.distributions.distributions.Distribution.ipdf`:

    .. math::

        \lambda = \frac{1}{k^2 + m^2} + \lambda_-

    where :math:`m^2` is the mass (inverse of the largest eigenvalue) and
    :math:`\lambda_-` is the smallest eigenvalue of the distribution.

    Parameters
    ----------
    data : dict[str, Any]
        The results of the computation of the canonical dimensions (as produced
        by the canonical-dimensions scripts).  Expected keys: ``"k2"``,
        ``"dimu2"``, ``"dimu4"``, ``"dimu6"``.
    dist : Distribution
        The distribution instance used to perform the computation.  Its
        ``m2`` and ``lminus`` attributes define the variable transformation,
        and its ``pdf`` method is used to draw the eigenvalue PDF on a twin
        axis.
    thresh : float
        The value of the threshold on the distribution to be considered
        "bulk". By default ``0.5``.
    suffix : str, optional
        The suffix of the output file name.
    analytic : bool
        Analytic computation. By default ``False``.
    output_dir : str | Path
        The output directory. By default ``"plots"``.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    # Convert k^2 to lam  using:  lam = 1 / (k^2 + m^2) + lam_-
    k2: Float[np.ndarray, n] = np.asarray(data["k2"])
    lam: Float[np.ndarray, n] = 1.0 / (k2 + dist.m2) + dist.lminus

    # Region of interest (computed in momentum space, index-compatible)
    idx: int
    start: int
    top: int
    idx, start, top = compute_roi(data=data, thresh=thresh)

    # Raw curves
    ax.plot(
        lam,
        data["dimu2"],
        "r-",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{2})$",
    )
    ax.plot(
        lam,
        data["dimu4"],
        "g--",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{4})$",
    )
    ax.plot(
        lam,
        data["dimu6"],
        "b-.",
        alpha=0.25 if not analytic else 1.0,
        label=None if not analytic else r"$\text{dim}(u_{6})$",
    )

    # Interpolations and annotations (non-analytic case)
    if not analytic:
        dimu2_interp: Callable
        dimu4_interp: Callable
        dimu6_interp: Callable
        dimu2_interp, dimu4_interp, dimu6_interp = interp_canonical_dimensions(
            data,
            idx,
        )

        # The interpolants are functions of k^2; evaluate at each λ point
        ax.plot(
            lam,
            dimu2_interp(k2),
            "r-",
            label=r"$\text{dim}(u_{2})$",
        )
        ax.plot(
            lam,
            dimu4_interp(k2),
            "g--",
            label=r"$\text{dim}(u_{4})$",
        )
        ax.plot(
            lam,
            dimu6_interp(k2),
            "b-.",
            label=r"$\text{dim}(u_{6})$",
        )

        # Shaded ROI (in lam coordinates; note: lam is *decreasing* in k^2,
        # so start/top indices are reversed on the lam axis)
        lam_start: float = lam[start]
        lam_top: float = lam[top]
        ax.axvspan(
            min(lam_start, lam_top),
            max(lam_start, lam_top),
            ls="dashed",
            color="r",
            alpha=0.05,
        )

        lam_idx: float = lam[idx]
        ax.axvline(lam_idx, ls="dashed", color="r", alpha=0.25)

        ax.plot([lam_idx], [dimu2_interp(k2[idx])], "ro")
        ax.text(
            x=lam_idx * 1.01,
            y=dimu2_interp(k2[idx]),
            s=rf"$\text{{dim}}(u_2) = {dimu2_interp(k2[idx]):.2f}$",
            color="r",
            fontsize=10,
            ha="left",
            va="bottom",
        )
        ax.plot([lam_idx], [dimu4_interp(k2[idx])], "go")
        ax.text(
            x=lam_idx * 1.01,
            y=dimu4_interp(k2[idx]),
            s=rf"$\text{{dim}}(u_4) = {dimu4_interp(k2[idx]):.2f}$",
            color="g",
            fontsize=10,
            ha="left",
            va="bottom",
        )
        ax.plot([lam_idx], [dimu6_interp(k2[idx])], "bo")
        ax.text(
            x=lam_idx * 1.01,
            y=dimu6_interp(k2[idx]),
            s=rf"$\text{{dim}}(u_6) = {dimu6_interp(k2[idx]):.2f}$",
            color="b",
            fontsize=10,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("canonical dimensions")
    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    _ratio: float | None = getattr(dist, "ratio", None)
    _var: float | None = getattr(dist, "var", None)
    _legend_title: str | None = (
        rf"$q = {_ratio},\ \sigma^2 = {_var}$"
        if _ratio is not None and _var is not None
        else None
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
        title=_legend_title,
    )

    # Twin axis: eigenvalue PDF  µ(λ) — plotted over the full spectrum [0, λ+]
    lam_full: Float[np.ndarray, n] = np.linspace(0.0, dist.lplus, num=1000)
    ax2 = ax.twinx()
    ax2.plot(lam_full, dist.pdf(lam_full), "k--")
    ax2.set(ylabel=r"$\mu$")

    if suffix is None:
        plt.savefig(out_dir / "canonical_dimensions_eigenvalues.pdf")
    else:
        plt.savefig(out_dir / f"canonical_dimensions_eigenvalues_{suffix}.pdf")


def _setup_figure(
    image: str | Path | None,
) -> tuple[Figure, Axes | dict[str, Axes]]:
    """Set up the figure for plotting.

    Returns
    -------
    tuple[Figure, Axes | dict[str, Axes]]
        The figure and axes objects.

    Raises
    ------
    FileNotFoundError
        If the provided image file does not exist.
    """
    if image is not None:
        image_path: Path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(
                f"The image you provided ({image_path!s}) could not be found!",
            )
        img_arr: np.ndarray = np.array(Image.open(image_path))
        fig: Figure = plt.figure(figsize=(9, 5), layout="constrained")
        axs = fig.subplot_mosaic(
            [["left", "top_right"], ["left", "."]],
            width_ratios=[2, 1],
        )
        # Plot the image
        axs["top_right"].grid(False)
        axs["top_right"].axis("off")
        axs["top_right"].imshow(img_arr)
        return fig, axs

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    return fig, ax


def _plot_dim_with_ema(
    ax: Axes,
    x: np.ndarray,
    dim: np.ndarray,
    label: str,
    color_style: str,
    win: int,
) -> None:
    """Plot a canonical dimension and its EMA if requested."""
    ax.plot(
        x,
        dim,
        color_style,
        alpha=1.0 if win <= 0 else 0.15,
        label=label if win <= 0 else None,
    )
    if win > 0:
        new_x: Float[np.ndarray, nV]
        new_dim: Float[np.ndarray, nV]
        new_x, new_dim = _ema(np.array(x), dim, win=win)
        ax.plot(
            new_x,
            new_dim,
            color_style,
            alpha=1.0,
            label=label,
        )


def plot_canonical_dimensions_scan(
    x: list[float] | Float[np.ndarray, n],
    name: str,
    win: int = 0,
    dimu2: Float[np.ndarray, n] | None = None,
    dimu4: Float[np.ndarray, n] | None = None,
    dimu6: Float[np.ndarray, n] | None = None,
    suffix: str | None = None,
    image: str | Path | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the canonical dimensions as a function of a particular quantity of interest.

    Parameters
    ----------
    x : array of floats
        The quantity of interest (shape: :math:`n`).
    name : str
        The label of the x-axis.
    win : int
        The smoothing window width. By default `0`.
    dimu2 : array of floats, optional
        The list of values of the canonical dimension of the quadratic coupling (shape: :math:`n`).
    dimu4 : array of floats, optional
        The list of values of the canonical dimension of the quartic coupling (shape: :math:`n`).
    dimu6 : array of floats, optional
        The list of values of the canonical dimension of the sextic coupling (shape: :math:`n`).
    suffix : str, optional
        The suffix to postpone to the file name.
    image : str | Path, optional
        The path to the image used for the computations.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax_setup: Axes | dict[str, Axes]
    fig, ax_setup = _setup_figure(image)
    ax: Axes = ax_setup["left"] if isinstance(ax_setup, dict) else ax_setup

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    x_arr = np.array(x)
    if dimu2 is not None:
        _plot_dim_with_ema(ax, x_arr, dimu2, r"$\text{dim}(u_{2})$", "r-", win)
    if dimu4 is not None:
        _plot_dim_with_ema(ax, x_arr, dimu4, r"$\text{dim}(u_{4})$", "g--", win)
    if dimu6 is not None:
        _plot_dim_with_ema(ax, x_arr, dimu6, r"$\text{dim}(u_{6})$", "b-.", win)

    ax.set(xlabel=name, ylabel="canonical dimensions")
    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    if suffix is None:
        plt.savefig(out_dir / "canonical_dimensions_snr.pdf")
    else:
        plt.savefig(out_dir / f"canonical_dimensions_{suffix}.pdf")


def plot_canonical_dimensions_scan2d(
    snr: Float[np.ndarray, n],
    lam: Float[np.ndarray, n],
    dimu2: Float[np.ndarray, n],
    dimu4: Float[np.ndarray, n],
    dimu6: Float[np.ndarray, n],
    suffix: str | None = None,
    image: str | None = None,
    output_dir: str | Path = "plots",
):
    """Plot the canonical dimensions as a surface plot of the signal-to-noise ratio and the Poisson parameter.

    Parameters
    ----------
    snr : array of floats
        The signal-to-noise ratio (shape: :math:`n`).
    lam : array of floats
        The Poisson parameter (shape: :math:`n`).
    dimu2 : array of floats, optional
        The list of values of the canonical dimension of the quadratic coupling (shape: :math:`n`).
    dimu4 : array of floats, optional
        The list of values of the canonical dimension of the quartic coupling (shape: :math:`n`).
    dimu6 : array of floats, optional
        The list of values of the canonical dimension of the sextic coupling (shape: :math:`n`).
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
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    if image is not None:
        image_path: Path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(
                f"The image you provided ({image_path!s}) could not be found!",
            )
        img_arr: np.ndarray = np.array(Image.open(image_path))
        fig: Figure = plt.figure(figsize=(24, 5), layout="constrained")
        ax = fig.subplot_mosaic(
            [
                ["left", "center", "right", "top_right"],
                ["left", "center", "right", "."],
            ],
            width_ratios=[2, 2, 2, 1],
        )

        # Plot the image
        ax["top_right"].grid(False)
        ax["top_right"].axis("off")
        ax["top_right"].imshow(img_arr)

        # Share y-axis among the three canonical-dimension panels
        ax["center"].sharey(ax["left"])
        ax["right"].sharey(ax["left"])
    else:
        fig, axes = plt.subplots(
            figsize=(21, 5),
            ncols=3,
            sharey=True,
            layout="constrained",
        )
        ax: dict[str, Axes] = {
            "left": axes[0],
            "center": axes[1],
            "right": axes[2],
        }

    ax["left"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax["left"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax["center"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax["center"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax["right"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax["right"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    # Handle nans
    dimu2: Float[np.ndarray, n] = np.nan_to_num(dimu2)
    dimu4: Float[np.ndarray, n] = np.nan_to_num(dimu4)
    dimu6: Float[np.ndarray, n] = np.nan_to_num(dimu6)

    # Shared color scale for direct comparison
    all_dims: np.ndarray = np.concatenate((dimu2, dimu4, dimu6))
    vmin: float = float(np.min(all_dims))
    vmax: float = float(np.max(all_dims))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    levels: np.ndarray = np.linspace(vmin, vmax, 21)

    # Surface plot
    ax["left"].tricontourf(
        snr,
        lam,
        dimu2,
        levels=levels,
        cmap="turbo",
        norm=norm,
    )
    ax["center"].tricontourf(
        snr,
        lam,
        dimu4,
        levels=levels,
        cmap="turbo",
        norm=norm,
    )
    cntr_right = ax["right"].tricontourf(
        snr,
        lam,
        dimu6,
        levels=levels,
        cmap="turbo",
        norm=norm,
    )
    fig.colorbar(
        cntr_right,
        ax=ax["right"],
        label="canonical dimension",
    )

    ax["left"].set_title(r"$\text{dim}(u_{2})$")
    ax["center"].set_title(r"$\text{dim}(u_{4})$")
    ax["right"].set_title(r"$\text{dim}(u_{6})$")

    # Set the labels
    ax["left"].set(
        xlabel="signal-to-noise ratio ($\beta$)",
        ylabel=r"Poisson expectation ($\lambda$)",
    )
    ax["left"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax["center"].set(
        xlabel="signal-to-noise ratio ($\beta$)",
        ylabel="",
    )
    ax["center"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax["right"].set(
        xlabel="signal-to-noise ratio ($\beta$)",
        ylabel="",
    )
    ax["right"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    if suffix is None:
        plt.savefig(out_dir / "canonical_dimensions2d_snr.pdf")
    else:
        plt.savefig(out_dir / f"canonical_dimensions2d_{suffix}.pdf")


def plot_ratio_scan(
    groups: pd.DataFrame,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the canonical dimensions as a function of a particular quantity of interest.

    Parameters
    ----------
    groups : pd.DataFrame
        The grouped dataframe containing the points to plot.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    ax.fill_between(
        groups.index,
        groups["dimu2", "mean"] - groups["dimu2", "std"],
        groups["dimu2", "mean"] + groups["dimu2", "std"],
        color="r",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups["dimu2", "mean"],
        "r-",
        label=r"$\text{dim}(u_{2})$",
    )
    ax.fill_between(
        groups.index,
        groups["dimu4", "mean"] - groups["dimu4", "std"],
        groups["dimu4", "mean"] + groups["dimu4", "std"],
        color="g",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups["dimu4", "mean"],
        "g--",
        label=r"$\text{dim}(u_{4})$",
    )
    ax.fill_between(
        groups.index,
        groups["dimu6", "mean"] - groups["dimu6", "std"],
        groups["dimu6", "mean"] + groups["dimu6", "std"],
        color="b",
        alpha=0.15,
    )
    ax.plot(
        groups.index,
        groups["dimu6", "mean"],
        "b-.",
        label=r"$\text{dim}(u_{6})$",
    )

    ax.set(xlabel="ratio ($q = p / n$)", ylabel="canonical dimensions")
    ax.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    if suffix is None:
        plt.savefig(out_dir / "canonical_dimensions_ratio.pdf")
    else:
        plt.savefig(out_dir / f"canonical_dimensions_ratio_{suffix}.pdf")


def plot_localization(
    data: dict[str, Any],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> tuple[float, float, float, float, float]:
    """Plot the localization of the components of the eigenvectors in the UV and IR.

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
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Find the corresponding index of the eigenvalues/vectors
    idx: int = int(np.argmin(np.abs(np.array(data["evl"]) - data["lplus_mp"])))

    # Consider 100 eigenvectors in the UV and the IR
    n_right: int = len(data["evl"]) - idx
    n_left: int = 100 - n_right

    evc_uv: Float[np.ndarray, 100, p] = np.array(data["evc"]).T[:100].ravel()
    evc_ir: Float[np.ndarray, 100, p] = (
        np.array(data["evc"]).T[idx - n_left : idx + n_right].ravel()
    )

    # Plot the results
    fig: Figure = plt.figure(figsize=(14, 5))
    axs = fig.subplot_mosaic(
        [["hist", "joint"], ["ratio", "joint"]],
        height_ratios=[2, 1],
        sharex=True,
    )

    axs["hist"].axes.get_xaxis().set_visible(False)
    axs["hist"].axhline(0.0, color="k", ls="dotted", alpha=0.15)
    axs["hist"].axvline(0.0, color="k", ls="dotted", alpha=0.15)

    uv_bins: np.ndarray
    uv_edges: np.ndarray
    uv_bins, uv_edges, _ = axs["hist"].hist(
        evc_uv,
        bins=2 * int(np.sqrt(len(data["evl"]))),
        color="k",
        label="UV",
        density=True,
        histtype="step",
    )
    axs["hist"].axvline(evc_uv.mean(), color="k", ls="dashed")

    ir_bins: np.ndarray
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
        ratio: np.ndarray = ir_bins / uv_bins

        axs["ratio"].plot(uv_edges[:-1], ratio, color="k")
        axs["ratio"].axhline(1.0, color="k", ls="dashed", alpha=0.15)
        axs["ratio"].axvline(0.0, color="k", ls="dashed", alpha=0.15)
        axs["ratio"].set_ylim(0.8, 1.2)
        axs["ratio"].set_yticks([0.9, 1.0, 1.1])

        axs["ratio"].set_ylabel("IR / UV")

    axs["joint"].hist2d(
        evc_ir,
        evc_uv,
        bins=uv_edges,
        density=True,
        cmap="turbo",
    )
    axs["joint"].yaxis.tick_right()
    axs["joint"].yaxis.set_label_position("right")
    axs["joint"].set(xlabel="IR", ylabel="UV")

    plt.subplots_adjust(hspace=0, wspace=0)
    if suffix is None:
        plt.savefig(out_dir / "localization_plot.pdf", dpi=300)
    else:
        plt.savefig(out_dir / f"localization_plot_{suffix}.pdf", dpi=300)

    # Find the point around the average value of the UV distribution
    res_idx: int = int(np.argmin(np.abs(uv_edges - evc_uv.mean())))

    return (
        float(ratio[res_idx]),
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
) -> None:
    """Plot the eigenvalues of the distribution.

    Parameters
    ----------
    data : dict[str, Any]
        The eigenvalues and eigenvectors.
    suffix : str
        The name to postpone to the file name.
    zoom : bool
        Add an inset axis to zoom on the IR region. By default `False`.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    evls: Float[np.ndarray, p] = np.array(data["evl"])
    ax.hist(
        evls,
        bins=2 * int(np.sqrt(len(evls))),
        color="k",
        density=True,
        histtype="step",
    )

    ax.axhline(0.0, ls="dashed", color="k", alpha=0.15)
    ax.axvline(0.0, ls="dashed", color="k", alpha=0.15)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\mu$")

    if zoom:
        max_evls: float = float(max(evls))
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
        plt.savefig(out_dir / "eigenvalues_dist.pdf")
    else:
        plt.savefig(out_dir / f"eigenvalues_dist_{suffix}.pdf")


def plot_localization_scan(
    snrs: list[float] | Float[np.ndarray, n],
    ratios: list[float] | Float[np.ndarray, n],
    uv_stds: list[float] | Float[np.ndarray, n],
    ir_means: list[float] | Float[np.ndarray, n],
    ir_stds: list[float] | Float[np.ndarray, n],
    output_dir: str | Path = "plots",
) -> None:
    """Plot values of the localization of the eigenvector components as a functions of the signal-to-noise ratio.

    Parameters
    ----------
    snrs : array of floats
        The list of the signal-to-noise ratio values (shape: :math:`n`).
    ratios : array of floats
        The list of ratios at the average of the UV distributions (shape: :math:`n`).
    uv_stds : array of floats
        The standard deviations of the UV distribution (shape: :math:`n`).
    ir_means : array of floats
        The mean values of the IR distribution (shape: :math:`n`).
    ir_stds : array of floats
        The standard deviations of the IR distribution (shape: :math:`n`).
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax: np.ndarray
    fig, ax = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=(7 * 2, 5 * 2),
        layout="constrained",
    )
    ax = ax.ravel()

    ax[0].plot(snrs, ratios, "kx")
    ax[0].plot(snrs, ratios, "k--", alpha=0.5)
    ax[0].set(xlabel="signal-to-noise ratio ($\beta$)", ylabel="ratio")
    ax[0].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax[0].axhline(1.0, ls="dashed", color="k", alpha=0.15)
    ax[1].plot(snrs, ir_means, "rx")
    ax[1].plot(snrs, ir_means, "r--", alpha=0.5)
    ax[1].set(xlabel="signal-to-noise ratio ($\beta$)", ylabel="IR (average)")
    ax[1].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax[2].plot(snrs, ir_stds, "rx")
    ax[2].plot(snrs, ir_stds, "r--", alpha=0.5)
    ax[2].set(
        xlabel="signal-to-noise ratio ($\beta$)",
        ylabel="IR (standard deviation)",
    )
    ax[2].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax[3].plot(snrs, np.array(ir_stds) / np.array(uv_stds), "kx")
    ax[3].plot(snrs, np.array(ir_stds) / np.array(uv_stds), "k--", alpha=0.5)
    ax[3].set(
        xlabel="signal-to-noise ratio ($\beta$)",
        ylabel="IR / UV (standard deviation)",
    )
    ax[3].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax[3].axhline(1.0, ls="dashed", color="k", alpha=0.15)

    plt.savefig(out_dir / "localization_scan.pdf")


def plot_trajectories(
    data: dict[str, Any],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the FRG trajectories.

    Parameters
    ----------
    data : dict[str, Any]
        The list of the signal-to-noise ratio values.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure = plt.figure(figsize=(14, 10), layout="constrained")
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
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
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
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
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
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
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
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    # Show trakectories
    axs["u2u4"].plot(data["u2"], data["u4"], "k-")
    axs["u2u4"].plot([data["u2"][0]], [data["u4"][0]], "bo", label="UV")
    axs["u2u4"].plot([data["u2"][-1]], [data["u4"][-1]], "ro", label="IR")
    axs["u2u4"].set(xlabel="$u_2$", ylabel="$u_4$")
    axs["u2u4"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u4"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u4"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    axs["u4u6"].plot(data["u4"], data["u6"], "k-")
    axs["u4u6"].plot([data["u4"][0]], [data["u6"][0]], "bo", label="UV")
    axs["u4u6"].plot([data["u4"][-1]], [data["u6"][-1]], "ro", label="IR")
    axs["u4u6"].set(xlabel="$u_4$", ylabel="$u_6$")
    axs["u4u6"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u4u6"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u4u6"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    axs["u4u6"].legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncols=1,
        frameon=False,
    )

    axs["u2u6"].plot(data["u2"], data["u6"], "k-")
    axs["u2u6"].plot([data["u2"][0]], [data["u6"][0]], "bo", label="UV")
    axs["u2u6"].plot([data["u2"][-1]], [data["u6"][-1]], "ro", label="IR")
    axs["u2u6"].set(xlabel="$u_2$", ylabel="$u_6$")
    axs["u2u6"].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u6"].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    axs["u2u6"].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    if suffix is None:
        plt.savefig(out_dir / "frg_equations.pdf")
    else:
        plt.savefig(out_dir / f"frg_equations_{suffix}.pdf")


def plot_symmetry_surface(
    phases_ir: list[int] | Integer[np.ndarray, n],
    u2: Float[np.ndarray, n],
    u4: Float[np.ndarray, n],
    u6: Float[np.ndarray, n],
    phases_uv: list[int] | Integer[np.ndarray, n] | None = None,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the symmetry surface of the FRG equations.

    Parameters
    ----------
    phases_ir : list[int] or array of ints
        The classification of the phase in the IR region (1 is symmetric, 0 is broken symmetry).
    u2 : array of floats
        The list of values of the quadratic coupling (shape: :math:`n`).
    u4 : array of floats
        The list of values of the quartic coupling (shape: :math:`n`).
    u6 : array of floats
        The list of values of the sextic coupling (shape: :math:`n`).
    phases_uv : list[int] or array of ints, optional
        The classification of the phase in the UV region (1 is symmetric, 0 is broken symmetry).
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Select only symmetric points in the UV
    if phases_uv is not None:
        phases_ir = np.array(phases_ir)[np.array(phases_uv).astype("bool")]
        u2 = np.array(u2)[np.array(phases_uv).astype("bool")]
        u4 = np.array(u4)[np.array(phases_uv).astype("bool")]
        u6 = np.array(u6)[np.array(phases_uv).astype("bool")]

    # Plot the points
    fig: Figure
    ax: np.ndarray
    fig, ax = plt.subplots(ncols=3, figsize=(21, 5), layout="constrained")

    col = [
        (1.0, 0.0, 0.0, 1.0) if p > 0 else (0.0, 0.0, 0.0, 0.15)
        for p in phases_ir
    ]
    ax[0].scatter(u2, u4, c=col)
    ax[0].set(xlabel="$u_2$", ylabel="$u_4$")
    ax[0].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[0].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[0].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    ax[1].scatter(u4, u6, c=col)
    ax[1].set(xlabel="$u_4$", ylabel="$u_6$")
    ax[1].axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[1].axvline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax[1].ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=(0.0, 0.0, 0.0, 0.15),
            marker="o",
            lw=0,
            label="broken symmetry",
        ),
        Line2D(
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
        axis="both",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    if suffix is None:
        plt.savefig(out_dir / "frg_symmetry_surface.pdf")
    else:
        plt.savefig(out_dir / f"frg_symmetry_surface_{suffix}.pdf")


def plot_symmetry_size(
    sizes: dict[str, float],
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the relative size of the symmetric phase.

    Parameters
    ----------
    sizes: dict[str, float]
        A collection containing the SNR as key and the relative size of the symmetric region as values.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    snr: np.ndarray = np.array(list(sizes.keys()))
    val: np.ndarray = 100 * np.array(list(sizes.values()))
    ax.plot(snr, val, "kx")
    ax.plot(snr, val, "k--", alpha=0.15)
    ax.set(xlabel="signal-to-noise ratio ($\beta$)", ylabel="relative size")
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.PercentFormatter(decimals=1, is_latex=True),
    )

    if suffix is None:
        plt.savefig(out_dir / "frg_symmetry_size.pdf")
    else:
        plt.savefig(out_dir / f"frg_symmetry_size_{suffix}.pdf")


def plot_potential(
    x: Float[np.ndarray, n],
    u2: dict[float, list[float]],
    u4: dict[float, list[float]],
    n: int,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Plot the potential.

    Parameters
    ----------
    x : array of floats
        The list of point to be evaluated (shape: :math:`n`).
    u2 : dict[float, list[float]]
        A collection of quadratic couplings per SNR.
    u4 : dict[float, list[float]]
        A collection of quartic couplings per SNR.
    n : int
        The index of the sample to consider.
    suffix : str
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    fig: Figure
    ax: np.ndarray
    fig, ax = plt.subplots(nrows=2, figsize=(9, 10), layout="constrained")

    ax[0].axhline(0.0, color="k", ls="dashed", alpha=0.15)
    ax[0].axvline(0.0, color="k", ls="dashed", alpha=0.15)
    ax[0].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    # Define the potential
    def _potential(
        x_vals: np.ndarray,
        u2_vals: dict[float, list[float]],
        u4_vals: dict[float, list[float]],
        snr_val: float,
        idx_val: int = 0,
    ) -> Float[np.ndarray, n]:
        y_vals: Float[np.ndarray, n] = (
            u2_vals[snr_val][idx_val] * x_vals**2
            + u4_vals[snr_val][idx_val] * x_vals**4
        )
        return y_vals - y_vals.min()

    # Plot the curves with different styles
    colors = mpl.colormaps["tab10"]
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
    y_max: float = -np.inf
    y_collection: Float[np.ndarray, SNR, n] = np.zeros((
        len(u2.keys()),
        len(x),
    ))
    for m, snr in enumerate(u2.keys()):
        y: np.ndarray = _potential(
            x,
            u2_vals=u2,
            u4_vals=u4,
            snr_val=snr,
            idx_val=n,
        )
        y_collection[m] = y
        y_max = float(np.maximum(y_max, y[np.argmin(np.abs(x))]))
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
    img: mpl.image.AxesImage = ax[1].imshow(
        y_collection.clip(0.0, 1.15 * y_max),
        aspect=50 * (7 / 5),
        cmap="viridis",
    )
    ax[1].set(xlabel="$M$", ylabel="signal-to-noise ratio ($\beta$)")
    ax[1].set_yticks(range(len(u2.keys())))
    ax[1].set_yticklabels([f"{val:.2f}" for val in u2])
    l_idx: int = int(np.argmin(np.abs(x + 1.0)))
    r_idx: int = int(np.argmin(np.abs(x - 1.0)))
    ax[1].set_xticks([l_idx, 500, r_idx])
    ax[1].set_xticklabels([f"{val:.2f}" for val in x[ax[1].get_xticks()]])
    formatter: mpl.ticker.ScalarFormatter = mpl.ticker.ScalarFormatter(
        useMathText=True,
    )
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar: mpl.colorbar.Colorbar = fig.colorbar(
        img,
        ax=ax[1],
        shrink=0.9,
        pad=-0.35,
        format=formatter,
    )
    cbar.set_label("$U$", labelpad=10)

    if suffix is None:
        plt.savefig(out_dir / "frg_potential.pdf")
    else:
        plt.savefig(out_dir / f"frg_potential_{suffix}.pdf")


def direct_relative_adherence(
    data: dict[str, Any],
    thresh: float = 0.5,
    suffix: str | None = None,
    output_dir: str | Path = "plots",
) -> None:
    """Compute the direct relative adherence.

    Parameters
    ----------
    data : dict[str, Any]
        The results of the computation of the canonical dimensions.
    thresh : float
        The value of the threshold on the distribution to be considered "bulk". By default `0.5`.
    suffix : str, optional
        The name to postpone to the file name.
    output_dir : str | Path
        The output directory. By default `"plots"`.
    """
    out_dir: Path = Path(output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Compute the point between the max and the start
    k2: Float[np.ndarray, n] = np.array(data["k2"])
    idx: int
    idx, _, _ = compute_roi(data=data, thresh=thresh)
    dimu4_interp: Callable
    _, dimu4_interp, _ = interp_canonical_dimensions(data, idx)
    dimu4_emp: Float[np.ndarray, n] = dimu4_interp(k2)

    # Compute the proxy
    ratios: Float[np.ndarray, 10] = np.linspace(0.10, 0.99, num=10)
    proxy: MarchenkoPastur = MarchenkoPastur(ratio=0.9, var=1.0)
    best_distance: float = np.inf
    best_dimu4: Float[np.ndarray, n] = np.zeros_like(k2)
    for ratio in ratios:
        sigma: float = float(np.sqrt(1 / (4.0 * np.sqrt(ratio)) / data["m2"]))
        mp: MarchenkoPastur = MarchenkoPastur(ratio=ratio, var=sigma)

        # Canonical dimensions
        dimu4: Float[np.ndarray, n]
        _, dimu4, _, _ = mp.canonical_dimensions(k2).T
        distance: float = float(np.abs(dimu4 - dimu4_emp).max())
        if distance < best_distance:
            best_distance = distance
            best_dimu4 = deepcopy(dimu4)
            proxy = deepcopy(mp)

    # Compute the adherence
    adherence: Float[np.ndarray, n] = np.zeros_like(k2)
    for n in range(1, len(k2)):
        adherence[n] = float((best_dimu4[:n] - dimu4_emp[:n]).min())

    # Plot
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    ax.axhline(0.0, color="k", alpha=0.15, linestyle="dashed")
    ax.axvline(0.0, color="k", alpha=0.15, linestyle="dashed")

    ax.plot(k2[1:], adherence[1:], "r-", label="local inverse adherence")
    ax.set(xlabel="$k^2$", ylabel=r"$\zeta^{-1}_{k^2}$")

    ax_twin: Axes = ax.twinx()

    slope_change: np.ndarray = np.abs(np.diff(adherence))
    idx_slope: int = int(slope_change.nonzero()[0][-1])
    ax.axvline(data["k2"][idx_slope], color="r", ls="dashed")
    pos: float = (adherence.max() - adherence.min()) / 2.0 + adherence.min()
    ax.text(
        data["k2"][idx_slope] * 0.99,
        ha="right",
        y=pos,
        s=f"$k^2_c$ = {data['k2'][idx_slope]:.2f}",
        color="r",
        rotation=90,
    )

    ax_twin.plot(k2, np.array(data["dist"]), "k-")
    full_k2: Float[np.ndarray, 1000] = np.linspace(0.0, k2[-1], num=1000)
    ax_twin.plot(full_k2, proxy.ipdf(full_k2), "k--", alpha=0.75)
    ax_twin.set_ylabel("PDF")

    # Legend
    custom_lines = [
        Line2D([0], [0], color="r"),
        Line2D([0], [0], color="k", ls="dashed", alpha=0.5),
        Line2D([0], [0], color="k"),
    ]

    ax.legend(
        handles=custom_lines,
        labels=[
            r"$\zeta^{-1}_{k^2}$",
            "proxy",
            "data",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        ncols=3,
    )

    if suffix is None:
        plt.savefig(out_dir / "adherence.pdf")
    else:
        plt.savefig(out_dir / f"adherence_{suffix}.pdf")
        plt.savefig(out_dir / f"adherence_{suffix}.pdf")

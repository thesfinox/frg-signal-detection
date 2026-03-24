"""
Test Analysis

Test the analysis utilities
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from frg.utils.analysis import (
    _ema,
    add_values,
    canonical_dimensions_argsort,
    canonical_dimensions_files,
    compute_roi,
    direct_relative_adherence,
    extract_interp_values,
    interp_canonical_dimensions,
    plot_canonical_dimensions,
    plot_distribution,
    plot_eigenvalues,
    plot_localization,
    plot_localization_scan,
    plot_potential,
    plot_ratio_scan,
    plot_symmetry_size,
    plot_symmetry_surface,
    plot_trajectories,
)


def test_ema():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    new_x, new_y = _ema(x, y, win=2)
    assert len(new_x) < len(x)


def test_compute_roi():
    k2 = np.linspace(0, 1, 100)
    dist_vals = np.exp(-((k2 - 0.2) ** 2))
    data = {"k2": k2.tolist(), "dist": dist_vals.tolist()}
    idx, start, top = compute_roi(data, thresh=0.1)
    assert isinstance(idx, int)


def test_interp_canonical_dimensions():
    k2 = np.linspace(0, 1, 100)
    data = {
        "k2": k2.tolist(),
        "dimu2": (k2 * 0.1).tolist(),
        "dimu4": (k2 * 0.2).tolist(),
        "dimu6": (k2 * 0.3).tolist(),
        "dist": np.exp(-((k2 - 0.2) ** 2)).tolist(),
    }
    idx, _, _ = compute_roi(data, thresh=0.1)
    f2, f4, f6 = interp_canonical_dimensions(data, idx=idx)
    assert f2(k2[idx]) == pytest.approx(data["dimu2"][idx], rel=1e-1)


def test_extract_interp_values():
    k2 = np.linspace(0, 1, 100)
    data = {
        "k2": k2.tolist(),
        "dimu2": (k2 * 0.1).tolist(),
        "dimu4": (k2 * 0.2).tolist(),
        "dimu6": (k2 * 0.3).tolist(),
        "dist": np.exp(-((k2 - 0.2) ** 2)).tolist(),
    }
    v2, v4, v6, _ = extract_interp_values(data, thresh=0.1)
    assert isinstance(v2, float)


def test_canonical_dimensions_argsort():
    x = [1.0, 0.5]
    d2 = [0.1, 0.2]
    d4 = [0.3, 0.4]
    d6 = [0.5, 0.6]
    nx, nd2, nd4, nd6 = canonical_dimensions_argsort(x, d2, d4, d6)
    assert nx[0] == 0.5


@patch("matplotlib.pyplot.savefig")
def test_plots(mock_savefig, tmp_path):
    k2 = np.linspace(0, 1, 100)
    data = {
        "k2": k2.tolist(),
        "dist": np.exp(-((k2 - 0.2) ** 2)).tolist(),
        "dimu2": (k2 * 0.1).tolist(),
        "dimu4": (k2 * 0.2).tolist(),
        "dimu6": (k2 * 0.3).tolist(),
        "m2": 1.0,
        "lplus_mp": 0.8,
        "evl": np.random.rand(200).tolist(),
        "evc": np.random.rand(200, 200).tolist(),
        "u2": np.random.rand(100).tolist(),
        "u4": np.random.rand(100).tolist(),
        "u6": np.random.rand(100).tolist(),
    }

    from frg.distributions.distributions import MarchenkoPastur

    mp = MarchenkoPastur(ratio=0.5, var=1.0)
    plot_distribution(mp, output_dir=tmp_path)
    assert mock_savefig.called

    plot_canonical_dimensions(data, output_dir=tmp_path)
    plot_eigenvalues(data, output_dir=tmp_path, zoom=True)
    plot_trajectories(data, output_dir=tmp_path)


def test_add_values():
    interp_values = (0.1, 0.2, 0.3, 0.4)
    scale, d2, d4, d6 = [], [], [], []
    add_values(interp_values, scale, d2, d4, d6)
    assert len(scale) == 1
    assert scale[0] == 0.1


@patch("matplotlib.pyplot.savefig")
def test_more_plots(mock_savefig, tmp_path):
    df = pd.DataFrame(
        {
            ("dimu2", "mean"): [0.1, 0.2],
            ("dimu2", "std"): [0.01, 0.02],
            ("dimu4", "mean"): [0.3, 0.4],
            ("dimu4", "std"): [0.03, 0.04],
            ("dimu6", "mean"): [0.5, 0.6],
            ("dimu6", "std"): [0.05, 0.06],
        },
        index=[0.5, 0.9],
    )
    plot_ratio_scan(df, output_dir=tmp_path)

    sizes = {0.1: 0.5, 0.5: 0.8}
    plot_symmetry_size(sizes, output_dir=tmp_path)

    x = np.linspace(-1, 1, 1000)
    u2 = {0.1: [0.1], 0.5: [0.2]}
    u4 = {0.1: [0.3], 0.5: [0.4]}
    plot_potential(x, u2, u4, n=0, output_dir=tmp_path)


def test_file_helpers(tmp_path):
    k2 = np.linspace(0, 1, 100)
    data = {
        "dist": np.exp(-((k2 - 0.2) ** 2)).tolist(),
        "k2": k2.tolist(),
        "dimu2": (k2 * 0.1).tolist(),
        "dimu4": (k2 * 0.2).tolist(),
        "dimu6": (k2 * 0.3).tolist(),
    }
    (tmp_path / "file_snr=1.0.json").write_text(json.dumps(data))
    files = canonical_dimensions_files(tmp_path)
    assert len(files) == 4

    (tmp_path / "data_ratio=0.5_seed=42.json").write_text(json.dumps(data))
    # r_files = canonical_dimensions_ratio_files(tmp_path)
    # assert len(r_files) == 1


@patch("matplotlib.pyplot.savefig")
def test_localization_plots(mock_savefig, tmp_path):
    data = {
        "evl": np.linspace(0, 1, 200).tolist(),
        "evc": np.random.randn(200, 200).tolist(),
        "lplus_mp": 0.8,
    }
    plot_localization(data, output_dir=tmp_path)
    plot_localization_scan(
        [0.1], [1.0], [0.1], [0.0], [0.1], output_dir=tmp_path
    )


@patch("matplotlib.pyplot.savefig")
def test_symmetry_plots(mock_savefig, tmp_path):
    plot_symmetry_surface(
        [1, 0], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], output_dir=tmp_path
    )


@patch("matplotlib.pyplot.savefig")
def test_adherence(mock_savefig, tmp_path):
    k2 = np.linspace(0.1, 1, 100)
    data = {
        "k2": k2.tolist(),
        "dist": np.exp(-((k2 - 0.5) ** 2)).tolist(),
        "dimu2": (k2 * 0.1).tolist(),
        "dimu4": (k2 * 0.2).tolist(),
        "dimu6": (k2 * 0.3).tolist(),
        "m2": 1.0,
    }
    direct_relative_adherence(data, output_dir=tmp_path)
    assert mock_savefig.called

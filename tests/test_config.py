"""Test configuration.

Test the configuration initialisation
"""

import pathlib

import pytest

from frg.distributions.distributions import EmpiricalDistribution
from frg.utils.utils import get_cfg_defaults, load_data


def test_cfg_defaults():
    """Test the default configuration."""
    cfg = get_cfg_defaults()

    assert hasattr(cfg, "DIST")
    assert cfg.DIST.NUM_SAMPLES == 1000
    assert cfg.DIST.VAR == 1.0
    assert cfg.DIST.RATIO == 0.5
    assert cfg.DIST.SEED == 42

    assert hasattr(cfg, "SIG")
    assert cfg.SIG.INPUT is None
    assert cfg.SIG.SNR == 0.0

    assert hasattr(cfg, "POT")
    assert cfg.POT.UV_SCALE == 1.0e-5
    assert cfg.POT.KAPPA_INIT == 1.0e-5
    assert cfg.POT.U2_INIT == 1.0e-5
    assert cfg.POT.U4_INIT == 1.0e-5
    assert cfg.POT.U6_INIT == 1.0e-5

    assert hasattr(cfg, "DATA")
    assert cfg.DATA.OUTPUT_DIR == "results"


def test_load_data():
    """Test the loading of data."""
    cfg = get_cfg_defaults()

    cfg["SIG"]["INPUT"] = "spam.png"
    with pytest.raises(FileNotFoundError):
        load_data(cfg)

    cfg["SIG"]["INPUT"] = "tests/data/mnist.png"
    dist = load_data(cfg)
    assert isinstance(dist, EmpiricalDistribution)

    import numpy as np

    np.random.seed(42)
    X_dummy = np.random.randn(100, 50)
    np.save("tests/data/dummy.npy", np.cov(X_dummy, rowvar=False))
    cfg["SIG"]["INPUT"] = "tests/data/dummy.npy"
    dist = load_data(cfg)
    assert isinstance(dist, EmpiricalDistribution)

    # Test non-2D covariance
    np.save("tests/data/dummy.npy", np.ones(10))
    with pytest.raises(ValueError):
        load_data(cfg)

    # Test non-square covariance
    np.save("tests/data/dummy.npy", np.ones((10, 5)))
    with pytest.raises(ValueError):
        load_data(cfg)

    pathlib.Path("tests/data/dummy.npy").unlink()

    cfg["SIG"]["INPUT"] = "spam.npy"
    with pytest.raises(FileNotFoundError):
        load_data(cfg)

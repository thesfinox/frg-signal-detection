"""
Test the  distributions

Test the Marchenko-Pastur distribution.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from simulation import MarchenkoPastur


class TestMarchenkoPastur:
    def test_init(self):
        # Assert raise if ratio <= 0
        with pytest.raises(ValueError):
            MarchenkoPastur(0.0, 1.0)
            MarchenkoPastur(-1.0, 1.0)

        # Assert warning if ratio >= 1
        with pytest.warns(UserWarning):
            MarchenkoPastur(1.0, 1.0)
            MarchenkoPastur(2.0, 1.0)

        # Assert values
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.ratio == 0.5
        assert mp.sigma == 1.0
        assert mp.lplus == 1.0**2 * (1.0 + np.sqrt(0.5)) ** 2
        assert mp.lminus == 1.0**2 * (1.0 - np.sqrt(0.5)) ** 2

    def test_pdf(self):
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.pdf(0.0) == 0.0
        assert mp.pdf(50.0) == 0.0
        assert isinstance(mp.pdf([0.0, 50.0]), np.ndarray)
        assert (mp.pdf([0.0, 50.0]) == np.array([0.0, 0.0])).all()
        assert mp.pdf(mp.lminus) == 0.0
        assert mp.pdf(mp.lplus) == 0.0
        assert quad(mp.pdf, mp.lminus, mp.lplus)[0] == pytest.approx(1.0)

    def test_cdf(self):
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.cdf(0.0) == 0.0
        assert mp.cdf(50.0) == 1.0
        assert isinstance(mp.cdf([0.0, 50.0]), np.ndarray)
        assert (mp.cdf([0.0, 50.0]) == np.array([0.0, 1.0])).all()
        assert mp.cdf(mp.lminus) == 0.0
        assert mp.cdf(mp.lplus) == 1.0
        x = np.linspace(mp.lminus, mp.lplus, 100)
        y = mp.cdf(x)
        assert all((y[1:] - y[:-1]) >= 0.0)  # test monotonic increasing

        # Test with different arguments
        assert mp.cdf(2.0, x0=0.0) == mp.cdf(2.0)

    def test_ipdf(self):
        mp = MarchenkoPastur(0.5, 1.0)
        assert mp.ipdf(0.0) == 0.0
        assert mp.ipdf(0.000001) > 0.0
        assert mp.ipdf(50.0) > 0.0
        assert isinstance(mp.ipdf([0.0, 50.0]), np.ndarray)

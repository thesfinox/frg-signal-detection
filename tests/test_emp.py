"""
Test the empirical distribution

Test functions and methods of the empirical distribution data class.
"""

import numpy as np
import pytest

from frg.distributions.distributions import EmpiricalDistribution
from frg.utils.utils import get_cfg_defaults


class TestEmpiricalDistribution:
    """Test the empirical distribution"""

    def test_init(self):
        """Test the constructor of the class"""
        # Assert raise if n_samples < 2
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=1)

        # Assert raise if sigma <= 0
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=2000, var=0.0)
            EmpiricalDistribution(n_samples=2000, var=-1.0)

        # Assert raise if ratio <= 0
        with pytest.raises(ValueError):
            EmpiricalDistribution(n_samples=2000, ratio=0.0)
            EmpiricalDistribution(n_samples=2000, ratio=-1.0)

        # Assert raise if poisson_lam < 0
        with pytest.raises(ValueError):
            EmpiricalDistribution(
                n_samples=2000, is_poisson=True, poisson_lam=-1.0
            )

        # Assert raise if bad poisson_centering
        with pytest.raises(ValueError):
            EmpiricalDistribution(
                n_samples=2000, is_poisson=True, poisson_centering="wrong"
            )

        # Warning if 2 < n_samples < 1000
        with pytest.warns(UserWarning):
            EmpiricalDistribution(n_samples=100)

        # Assert values
        emp = EmpiricalDistribution(
            n_samples=1532,
            var=1.3,
            ratio=0.1,
            seed=123,
        )
        assert emp.n_samples == 1532
        assert emp.var == 1.3
        assert emp.ratio == 0.1
        assert emp.n_vars == int(1532 * 0.1)
        assert emp.seed == 123
        assert hasattr(emp, "data")
        assert isinstance(emp.data, np.ndarray)
        assert emp.data.shape == (1532, int(1532 * 0.1))
        assert emp.data.dtype == np.float64
        assert emp.data.mean() == pytest.approx(0.0, abs=1.0e-2)
        assert emp.data.var() == pytest.approx(1.3, abs=2.0e-1)

        with pytest.raises(ValueError):
            emp.data = np.ones(5)  # not 2D

    def test_poisson_logic(self):
        """Test the generation of Poisson noise."""
        # Case 1: Poisson data centered
        emp = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=True,
            poisson_lam=10.0,
            poisson_centering="centered",
        )
        assert emp.var == 10.0

        # Case 1: Poisson data non-centered
        emp_nc = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=True,
            poisson_lam=10.0,
            poisson_centering="non-centered",
        )
        assert emp_nc.var == 10.0

        # Case 1: Poisson data mirrored
        emp_m = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=True,
            poisson_lam=10.0,
            poisson_centering="mirrored",
        )
        assert emp_m.var == 20.0

        # Case 2: Poisson covariance centered
        emp_c_cov = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=False,
            poisson_lam=10.0,
            poisson_centering="centered",
        )
        assert hasattr(emp_c_cov, "_cov_noise")

        # Case 2: Poisson covariance non-centered
        emp_nc_cov = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=False,
            poisson_lam=10.0,
            poisson_centering="non-centered",
        )
        assert hasattr(emp_nc_cov, "_cov_noise")

        # Case 2: Poisson covariance mirrored
        emp_m_cov = EmpiricalDistribution(
            n_samples=1000,
            ratio=0.1,
            is_poisson=True,
            poisson_data=False,
            poisson_lam=10.0,
            poisson_centering="mirrored",
        )
        assert hasattr(emp_m_cov, "_cov_noise")
        # Call fit to hit line 1070
        emp_m_cov.fit()

    def test_unfitted_exceptions(self):
        """Test exceptions raised when calling methods before fitting."""
        emp = EmpiricalDistribution(n_samples=1000)
        with pytest.raises(ValueError):
            emp.pdf(1.0)
        with pytest.raises(ValueError):
            emp.cdf(1.0)
        with pytest.raises(ValueError):
            emp.icdf(1.0)

    def test_from_config(self):
        """Test the from_config method of the class"""
        cfg = get_cfg_defaults()
        emp = EmpiricalDistribution.from_config(cfg)
        assert emp.n_samples == 1000
        assert emp.var == 1.0
        assert emp.ratio == 0.5
        assert emp.n_vars == int(1000 * 0.5)
        assert emp.seed == 42

    def test_from_covariance(self):
        """Test the from_covariance method of the class"""
        cfg = get_cfg_defaults()
        X = np.random.randn(100, 50)
        cov = np.cov(X, rowvar=False)
        assert cov.shape == (50, 50)
        emp = EmpiricalDistribution.from_covariance(cov, cfg=cfg)
        assert emp._iscov
        assert emp.n_samples == 1000
        assert emp.var == 1.0
        assert emp.ratio == 0.5
        assert emp.seed == 42
        assert (emp.data == cov).all()

    def test_add_signal(self):
        """Test the add_signal method of the class"""
        emp = EmpiricalDistribution(n_samples=1024, var=1.0, ratio=0.5)
        X = np.random.randn(1024, 512)

        with pytest.raises(ValueError):
            emp.add_signal(np.ones(10))

        emp2 = emp.add_signal(X)
        assert isinstance(emp2, EmpiricalDistribution)
        assert emp2.n_samples == 1024
        assert emp2.var == 1.0
        assert emp2.ratio == 0.5
        assert emp2.n_vars == int(1024 * 0.5)
        assert emp2.seed == 42
        assert (emp2.data == emp.data).all()
        emp3 = emp2.add_signal(X, snr=1.0)
        assert isinstance(emp3, EmpiricalDistribution)
        assert emp3.n_samples == 1024
        assert emp3.var == 1.0
        assert emp3.ratio == 0.5
        assert emp3.n_vars == int(1024 * 0.5)
        assert emp3.seed == 42
        assert (emp3.data == emp2.data).all()

        emp4 = emp2.add_signal(X, snr=-1.0)
        assert (emp4.data == X).all()

        # Warn if from covariance
        cfg = get_cfg_defaults()
        cov = np.cov(X, rowvar=False)
        emp_cov = EmpiricalDistribution.from_covariance(cov, cfg=cfg)
        with pytest.warns(UserWarning):
            emp_cov.add_signal(X, snr=1.0)

    def test_fit(self):
        """Test the fit method of the class"""
        emp = EmpiricalDistribution(n_samples=1024, var=1.0, ratio=0.5)
        X = np.random.randn(1024, 512)
        emp.fit(X, snr=0.5)
        assert hasattr(emp, "eigenvalues")
        assert isinstance(emp.eigenvalues, np.ndarray)
        assert hasattr(emp, "ipdf")
        assert hasattr(emp, "dipdf")
        assert hasattr(emp, "icdf")

        # PDF / CDF bounds testing
        assert emp.pdf(emp.lminus - 1.0) == 0.0
        assert emp.pdf(emp.lplus + 1.0) == 0.0
        # Test inside the support to hit line 1254
        assert emp.pdf((emp.lminus + emp.lplus) / 2.0) > 0.0

        assert emp.cdf(emp.lminus - 1.0) == 0.0
        assert emp.cdf(emp.lplus + 1.0) == 1.0

        assert isinstance(
            emp.pdf(np.array([emp.lminus - 1.0, emp.lplus + 1.0])), np.ndarray
        )
        assert isinstance(
            emp.cdf(np.array([emp.lminus - 1.0, emp.lplus + 1.0])), np.ndarray
        )

        # ICDF testing
        assert emp.icdf(0.0) == 0.0
        assert isinstance(emp.icdf(np.array([0.0, 1.0])), np.ndarray)

        # Test from the covariance matrix
        cfg = get_cfg_defaults()
        cov = np.cov(X, rowvar=False)
        emp2 = EmpiricalDistribution.from_covariance(cov, cfg=cfg)
        emp2.fit()

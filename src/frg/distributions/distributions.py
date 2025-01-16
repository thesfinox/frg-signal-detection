"""
Distributions used in experiments and simulations.
"""

from collections.abc import Callable
from typing import Self
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from skimage.transform import resize
from yacs.config import CfgNode


class Distribution:
    """Abstract base class for distributions"""

    def ipdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """Compute the PDF of the inverse distribution."""
        raise NotImplementedError

    def icdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """Compute the CDF of the inverse distribution."""
        raise NotImplementedError

    def dipdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """Compute the derivative of the PDF of the inverse distribution."""
        raise NotImplementedError

    def canonical_dimensions(self, x: float) -> ArrayLike:
        """
        Compute the canonical dimensions of the distribution.

        Returns
        -------
        ArrayLike
            An array containing, in columns:

            - ``dim_u2``: the mass term
            - ``dim_u4``: the quartic interaction
            - ``dim_u6``: the sextic interaction
            - ``dim_chi``: the dimension of the field
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(
                self.canonical_dimensions,
                otypes=[np.float64],
                signature="()->(n)",
            )(x)

        # Ignore the warning of zero value division and compute the dimensions
        with np.errstate(divide="ignore", invalid="ignore"):
            dimu2 = self.icdf(x) / self.ipdf(x) / x
            dimu4 = dimu2 * (3.0 + x * self.dipdf(x) / self.ipdf(x)) - 2.0
            dimu6 = -dimu2 + 2.0 * dimu4
            dimchi = 1.0 - dimu2

        return np.array([dimu2, dimu4, dimu6, dimchi])

    def _frg_equations_single(
        self,
        x: float,
        u2: float,
        u4: float,
        u6: float,
        dx: float = 1.0e-9,
    ) -> ArrayLike:
        """
        A single step of FRG equations.

        Parameters
        ----------
        x : float
            The momentum scale :math:`k^2` at which to start computing the FRG equations. This represents an energy scale in the UV region. The FRG computation ends in the IR region.
        u2 : float
            The initial value of the coupling :math:`u_2` at energy scale :math:`x`.
        u4 : float
            The initial value of the coupling :math:`u_4` at energy scale :math:`x`.
        u6 : float
            The initial value of the coupling :math:`u_6` at energy scale :math:`x`.
        dx : float
            The step size, by default 1.0e-9.

        Returns
        -------
        ArrayLike
            An array containing, in columns:

            - ``u2``: the running of the coupling :math:`u_2`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the prefactor
        pref = dx * self.ipdf(x) / self.icdf(x)

        # Canonical dimensions
        dimu2, dimu4, dimu6, _ = self.canonical_dimensions(x).T

        # Right Hand Sides
        u2_rhs = -dimu2 * u2 - 2.0 * u4 / (1.0 + u2) ** 2
        u4_rhs = (
            -dimu4 * u4
            - 2.0 * u6 / (1.0 + u2) ** 2
            + 12.0 * u4**2 / (1.0 + u2) ** 3
        )
        u6_rhs = (
            -dimu6 * u6
            + 60.0 * u4 * u6 / (1.0 + u2) ** 3
            - 108.0 * u6**3 / (1.0 + u2) ** 4
        )

        # Equations
        u2 -= pref * u2_rhs
        u4 -= pref * u4_rhs
        u6 -= pref * u6_rhs

        return np.array([u2, u4, u6])

    def frg_equations(
        self,
        x: float,
        u2_init: float,
        u4_init: float,
        u6_init: float,
        dx: float = 1.0e-9,
    ) -> ArrayLike:
        """
        Compute the FRG equations.

        Given initial conditions of the couplings :math:`u_2`, :math:`u_4`, and :math:`u_6` at a given momentum scale :math:`k^2`, the function returns the values of the couplings at the scale :math:`k^2 - \\Delta_k`. That is, the function returns the running of the couplings from the UV region towards the IR (:math:`k^2 \\to 0`).

        Parameters
        ----------
        x : float
            The momentum scale :math:`k^2` at which to start computing the FRG equations. This represents an energy scale in the UV region. The FRG computation ends in the IR region.
        u2_init : float
            The initial value of the coupling :math:`u_2` at energy scale :math:`x`.
        u4_init : float
            The initial value of the coupling :math:`u_4` at energy scale :math:`x`.
        u6_init : float
            The initial value of the coupling :math:`u_6` at energy scale :math:`x`.
        dx : float
            The step size, by default 1.0e-9.

        Returns
        -------
        ArrayLike
            An array containing, in columns:

            - ``k2``: the energy scale corresponding to the couplings
            - ``u2``: the running of the coupling :math:`u_2`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the energy scale
        k2, u2, u4, u6 = x, u2_init, u4_init, u6_init
        results = []
        while k2 >= dx:
            u2, u4, u6 = self._frg_equations_single(
                x=k2,
                u2=u2,
                u4=u4,
                u6=u6,
                dx=dx,
            )
            results.append([k2 - dx, u2, u4, u6])
            k2 -= dx
        return np.array(results)

    def _frg_equations_lpa_single(
        self,
        x: float,
        kappa: float,
        u4: float,
        u6: float,
        dx: float = 1.0e-9,
    ) -> ArrayLike:
        """
        A single step of FRG equations in Local Potential Approximation.

        Parameters
        ----------
        x : float
            The momentum scale :math:`k^2` at which to start computing the FRG equations. This represents an energy scale in the UV region. The FRG computation ends in the IR region.
        kappa : float
            The initial value of the position of the field :math:`\\chi` at energy scale :math:`x`.
        u4 : float
            The initial value of the coupling :math:`u_4` at energy scale :math:`x`.
        u6 : float
            The initial value of the coupling :math:`u_6` at energy scale :math:`x`.
        dx : float
            The step size, by default 1.0e-9.

        Returns
        -------
        ArrayLike
            An array containing, in columns:

            - ``kappa``: the running of the position of the zero :math:`kappa`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the prefactor
        pref = dx * self.ipdf(x) / self.icdf(x)

        # Canonical dimensions
        _, dimu4, dimu6, dimchi = self.canonical_dimensions(x).T

        # Right Hand Sides
        kappa_rhs = (
            -dimchi * kappa
            + 2.0
            * (3.0 + 2.0 * kappa * u6 / u4)
            / (1.0 + 2.0 * kappa * u4) ** 2
        )
        u4_rhs = (
            -dimu4 * u4
            + dimchi * kappa * u6
            - 10.0 * u6 / (1.0 + 2.0 * kappa * u4) ** 2
            + 4.0
            * (3.0 * u4 + 2.0 * kappa * u6) ** 2
            / (1.0 + 2.0 * kappa * u4) ** 3
        )
        u6_rhs = (
            -dimu6 * u6
            - 12.0
            * (3.0 * u4 + 2.0 * kappa * u6) ** 3
            / (1.0 + 2.0 * kappa * u4) ** 4
            + 40.0
            * u6
            * (3.0 * u4 + 2.0 * kappa * u6)
            / (1.0 + 2.0 * kappa * u4)
        )

        # Equations
        kappa -= pref * kappa_rhs
        u4 -= pref * u4_rhs
        u6 -= pref * u6_rhs

        return np.array([kappa, u4, u6])

    def frg_equations_lpa(
        self,
        x: float,
        kappa_init: float,
        u4_init: float,
        u6_init: float,
        dx: float = 1.0e-9,
    ) -> ArrayLike:
        """
        Compute the FRG equations in Local Potential Approximation.

        Given initial conditions of the zero of the potential :math:`\\kappa` and the couplings :math:`u_2`, and :math:`u_4` at a given momentum scale :math:`k^2`, the function returns the values of the couplings at the scale :math:`k^2 - \\Delta_k`. That is, the function returns the running of the couplings from the UV region towards the IR (:math:`k^2 \\to 0`).

        Parameters
        ----------
        x : float
            The momentum scale :math:`k^2` at which to start computing the FRG equations. This represents an energy scale in the UV region. The FRG computation ends in the IR region.
        kappa_init : float
            The initial value of the positin of the zero of the potential :math:`\\kappa` at energy scale :math:`x`.
        u4_init : float
            The initial value of the coupling :math:`u_4` at energy scale :math:`x`.
        u6_init : float
            The initial value of the coupling :math:`u_6` at energy scale :math:`x`.
        dx : float
            The step size, by default 1.0e-9.

        Returns
        -------
        ArrayLike
            An array containing, in columns:

            - ``k2``: the energy scale corresponding to the couplings
            - ``kappa``: the running of the zero of the potential :math:`kappa`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the energy scale
        k2, kappa, u4, u6 = x, kappa_init, u4_init, u6_init
        results = []
        while k2 >= dx:
            kappa, u4, u6 = self._frg_equations_lpa_single(
                x=k2,
                kappa=kappa,
                u4=u4,
                u6=u6,
                dx=dx,
            )
            results.append([k2 - dx, kappa, u4, u6])
            k2 -= dx
        return np.array(results)


class EmpiricalDistribution(Distribution):
    """Empitical signal-noise distribution"""

    def __init__(
        self,
        n_samples: int,
        sigma: float = 1.0,
        ratio: float = 0.5,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        n_samples : int
            The number of samples in the distribution.
        sigma : float
            The standard deviation of the distribution.
        ratio : float
            The ratio between the number of variables and the dimension of the sample/population.

            .. note::

                Let :math:`X \\in \\mathbb{R}^{n \\times p}` be a real matrix where samples are **row** vectors, then the ``ratio`` parameter is :math:`\\frac{p}{n}`.

        seed : int
            The random seed.

        Raises
        ------
        ValueError
            If the size of the sample is less than 2
        ValueError
            If sigma is not strictly positive
        ValueError
            If ratio is not strictly positive
        UserWarning
            If the size of the sample is less than 1000 (sample too small)
        """
        if n_samples < 2:
            raise ValueError(
                "The number of samples must be a large number but got %d < 2!"
                % n_samples
            )
        if n_samples < 1000:
            warn(
                "The number of samples is low (%d) and the results might suffer from numerical instabilities"
                % n_samples,
                category=UserWarning,
            )
        self.n_samples = int(n_samples)
        if sigma <= 0.0:
            raise ValueError(
                "The standard deviation must be strictly positive but got %f <= 0"
                % sigma
            )
        self.sigma = sigma
        if ratio <= 0.0:
            raise ValueError(
                "Ratio must be strictly positive but got %f <= 0" % ratio
            )
        self.ratio = ratio
        self.n_vars = int(n_samples * ratio)
        self.seed = int(seed)
        self._fitted = False

        # Generate the background distribution
        gen = np.random.default_rng(self.seed)
        self._iscov = False  # if generated directly from covariance
        self.data = gen.normal(
            loc=0.0,
            scale=self.sigma,
            size=(self.n_samples, self.n_vars),
        )

    @classmethod
    def from_config(cls, cfg: CfgNode) -> Self:
        """
        Create an instance of the class from a configuration.

        Parameters
        ----------
        cfg : CfgNode
            The configuration.

        Returns
        -------
        EmpiricalDistribution
            An instance of the class
        """
        return cls(
            n_samples=cfg.DIST.NUM_SAMPLES,
            sigma=cfg.DIST.SIGMA,
            ratio=cfg.DIST.RATIO,
            seed=cfg.DIST.SEED,
        )

    @classmethod
    def from_covariance(cls, cov: ArrayLike) -> Self:
        """
        Create an instance of the class from a covariance matrix.

        Parameters
        ----------
        cov : ArrayLike
            The covariance matrix.

        Returns
        -------
        EmpiricalDistribution
            An instance of the class
        """
        instance = cls(
            n_samples=9999,
            sigma=1.0,
            ratio=1.0,
            seed=9999,
        )
        instance.data = cov
        instance._iscov = True
        return instance

    def add_signal(self, X: ArrayLike, snr: float = 0.0) -> Self:
        """
        Add a signal to the distribution.

        Parameters
        ----------
        X : ArrayLike
            The signal to add.
        snr : float
            The signal-to-noise ratio.

        Returns
        -------
        EmpiricalDistribution
            A new instance of the class with the signal added.

        Raises
        ------
        ValueError
            If the signal-to-noise ratio is not positive
        """
        if snr < 0.0:
            raise ValueError(
                "The signal-to-noise ratio must be positive but got %f <= 0"
                % snr
            )

        # Add the signal to the background
        if snr > 0.0:
            self.data += snr * resize(X, output_shape=self.data.shape[:2])

        return self

    def _evl(self, X: ArrayLike) -> ArrayLike:
        """
        Get the eigenvalues of the covariance matrix of the data (from the singular values of the data).

        Parameters
        ----------
        X : ArrayLike
            The data.

        Returns
        -------
        ArrayLike
            The eigenvalues of the distribution.
        """
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        return S.ravel() ** 2 / (self.n_samples - 1)

    def _evl_cov(self, cov: ArrayLike) -> ArrayLike:
        """
        Get the eigenvalues of the covariance matrix.

        Parameters
        ----------
        cov : ArrayLike
            The covariance matrix.

        Returns
        -------
        ArrayLike
            The eigenvalues of the distribution.
        """
        return np.linalg.eigvalsh(cov)

    @property
    def eigenvalues_(self) -> ArrayLike:
        """
        Compute the eigenvalues of the distribution.

        .. note::

            This is the complete list of eigenvalues (bulk + spikes).
            You can access the filtered distribution **without** the spikes using the ``self.eigenvalues`` attribute, available after calling the ``self.fit(...)`` method.

        Returns
        -------
        ArrayLike
            The eigenvalues of the distribution, sorted in ascending order.
        """
        eigenvalues = (
            self._evl_cov(self.data) if self._iscov else self._evl(self.data)
        )
        return np.sort(eigenvalues)

    def _find_spikes(self, eigenvalues: ArrayLike) -> ArrayLike:
        """
        Find the spikes in the eigenvalues.

        Parameters
        ----------
        eigenvalues : ArrayLike
            The eigenvalues of the distribution.

        Returns
        -------
        ArrayLike
            The indices of the spikes.
        """
        dx = 1 / np.sqrt(len(eigenvalues))

        # Find the index of the beginning of the bulk distribution
        #
        #   >>> Going from right to left until the difference is smaller than dx
        diff = np.diff(eigenvalues)
        idx = np.argmin((diff > dx)[::-1])  # True if spike, False if bulk

        return len(eigenvalues) - int(idx)

    def fit(self, X: ArrayLike | None = None, snr: float = 0.0) -> Self:
        """
        Add the signal (if provided) and compute the eigenvalue distribution.

        Parameters
        ----------
        X : ArrayLike
            The signal to add.
        snr : float
            The signal-to-noise ratio.

        Returns
        -------
        EmpiricalDistribution
            A new instance of the class with the signal added and the eigenvalue distribution computed.
        """
        if X is not None:
            self.add_signal(X, snr=snr)

        # Remove the spikes from the eigenvalues
        eigenvalues = self.eigenvalues_
        self.eigenvalues = eigenvalues[: self._find_spikes(eigenvalues)]

        # Compute the momenta
        self.momenta = 1.0 / self.eigenvalues
        self.momenta -= np.min(self.momenta)  # shift to zero

        # Create a histogram
        dx = 1.0 / np.sqrt(self.data.shape[0])
        dens, edges = np.histogram(
            self.momenta,
            bins=np.arange(0.0, np.max(self.momenta) + dx, dx),
            density=True,
        )

        # Create the PDF
        self._ipdf = InterpolatedUnivariateSpline(
            edges[:-1] + dx / 2.0,
            dens,
            k=1,
            ext="zeros",
        )
        self._dipdf = self._ipdf.derivative()
        self._icdf = self._ipdf.antiderivative()

        self._fitted = True
        return self

    def ipdf(self, x: float | ArrayLike) -> ArrayLike:
        """
        Compute the PDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        ArrayLike
            The value(s) of the PDF at the given value(s).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling ipdf! Please call ``self.fit()`` first."
            )
        return self._ipdf(x)

    def dipdf(self, x: float | ArrayLike) -> ArrayLike:
        """
        Compute the derivative of the PDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the derivative of the PDF.

        Returns
        -------
        ArrayLike
            The value(s) of the derivative of the PDF at the given value(s).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling dipdf! Please call ``self.fit()`` first."
            )
        return self._dipdf(x)

    def icdf(self, x: float | ArrayLike) -> ArrayLike:
        """
        Compute the CDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        ArrayLike
            The value(s) of the CDF at the given value(s).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling icdf! Please call ``self.fit()`` first."
            )
        if np.isscalar(x):
            if x <= 0.0:
                return 0.0
            if x >= max(self.momenta):
                return 1.0
            return self._icdf(x)
        else:
            icdf = self._icdf(x)
            icdf[x <= 0.0] = 0.0
            icdf[x >= max(self.momenta)] = 1.0
            return icdf


class MarchenkoPastur(Distribution):
    """Marchenko-Pastur distribution"""

    def __init__(self, ratio: float, sigma: float):
        """
        Parameters
        ----------
        ratio : float
            The ratio between the number of variables and the dimension of the sample/population.

            .. note::

                Let :math:`X \\in \\mathbb{R}^{n \\times p}` be a real matrix where samples are **row** vectors, then the ``ratio`` parameter is :math:`\\frac{p}{n}`.

        sigma : float
            The standard deviation of the distribution.

        Raises
        ------
        ValueError
            If ratio is not strictly positive
        ValueError
            If the lowest eigenvalue is higher than the highest one
        UserWarning
            If ratio is higher than 1 because of numerical instabilities
        """
        # Sanitise the input
        if ratio <= 0.0:
            raise ValueError(
                "Ratio must be strictly positive but got %f <= 0" % ratio
            )
        if ratio >= 1.0:
            warn(
                "Ratios higher than 1 may result in numerical instabilities. We got %f >= 1"
                % ratio,
                category=UserWarning,
            )

        # Store the parameters
        self.ratio = ratio
        self.sigma = sigma

        # Compute the highest and lowest eigenvalues
        self.lplus = self.sigma**2 * (1.0 + np.sqrt(self.ratio)) ** 2
        self.lminus = self.sigma**2 * (1.0 - np.sqrt(self.ratio)) ** 2
        self.m2 = 1 / (self.lplus - self.lminus)
        if self.lminus > self.lplus:
            raise ValueError(
                "The smallest eigenvalue must be lower than the largest one, but got lminus = %f > %f = lplus!"
                % (self.lminus, self.lplus)
            )

    def pdf(
        self,
        x: float | ArrayLike,
        lminus: float | None = None,
        lplus: float | None = None,
    ) -> float | ArrayLike:
        """
        Compute the PDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the PDF.
        lminus : Optional[float], optional
            The lowest eigenvalue, by default None. Use the default value if left unspecified.
        lplus : Optional[float], optional
            The highest eigenvalue, by default None. Use the default value if left unspecified.

        Returns
        -------
        float | ArrayLike
            The value(s) of the PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.pdf, otypes=[np.float64])(x)

        lplus = self.lplus if lplus is None else lplus
        lminus = self.lminus if lminus is None else lminus
        if (x <= lminus) or (x >= lplus):
            return 0.0

        num = np.sqrt((lplus - x) * (x - lminus))
        den = 2.0 * np.pi * self.sigma**2 * self.ratio * x
        return num / den if den != 0.0 else 0.0

    def cdf(self, x: float | ArrayLike, x0: float = 0.0) -> float | ArrayLike:
        """
        Compute the CDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the CDF.
        x0 : float, optional
            The lower bound of integration, by default 0.

        Returns
        -------
        float | ArrayLike
            The value(s) of the CDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.cdf, otypes=[np.float64])(x)

        if x <= self.lminus:
            return 0.0
        if x >= self.lplus:
            return 1.0
        if x0 <= self.lminus:
            x0 = self.lminus
        return quad(lambda y: self.pdf(y), x0, x)[0]

    def _diff(self, func: Callable, x: float, eps: float) -> float:
        """
        Compute a derivative.

        Parameters
        ----------
        func: Callable
            The function to evaluate.
        x : float
            The value at which to evaluate the derivative.
        eps : float
            The value of the small variation, by default 1.0e-9.

        Returns
        -------
        float
            The value of the derivative of the PDF at the given value.
        """
        num = func(x + eps) - func(x - eps)
        den = 2.0 * eps
        return num / den if den != 0.0 else 0.0

    def dpdf(
        self,
        x: float | ArrayLike,
        eps: float = 1.0e-10,
    ) -> float | ArrayLike:
        """
        Compute the derivative of the PDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the derivative.
        eps : float
            The value of the small variation, by default 1.0e-10.

        Returns
        -------
        float | ArrayLike
            The value(s) of the derivative of the PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.dpdf, otypes=[np.float64])(x)
        return self._diff(self.pdf, x, eps)

    def ipdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """
        Compute the PDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the PDF.

        Returns
        -------
        float | ArrayLike
            The value(s) of the PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.ipdf, otypes=[np.float64])(x)

        # In order to reproduce the momentum distribution, we need to consider
        # a change of variable:
        #
        # 1) we consider lambda' = lambda - self.lminus,
        #
        # where lambda is an eigenvalue.
        #
        # This first change is to ensure that the distribution is shifted to
        # the origin (i.e. the lowest eigenvalue is zero).
        #
        # We then consider the change of variable:
        #
        # 2) lambda' = 1 / (k^2 + m^2),
        #
        # in order to switch to momenta (shifted by the mass m^2, which is the
        # inverse of the largest eigenvalue).
        return self.pdf(1 / (x + self.m2) + self.lminus) / (x + self.m2) ** 2

    def icdf(self, x: float | ArrayLike) -> float | ArrayLike:
        """
        Compute the CDF of the inverse Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        float | ArrayLike
            The value(s) of the CDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.icdf, otypes=[np.float64])(x)

        if x <= 0.0:
            return 0.0
        return quad(self.ipdf, 0.0, x)[0]

    def dipdf(
        self, x: float | ArrayLike, eps: float = 1.0e-10
    ) -> float | ArrayLike:
        """
        Compute the derivative of the inverse PDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | ArrayLike
            The value(s) at which to evaluate the derivative.
        eps : float
            The value of the small variation, by default 1.0e-10.

        Returns
        -------
        float | ArrayLike
            The value(s) of the derivative of the inverse PDF at the given value(s).
        """
        # If x is not a scalar, then vectorize the function
        if not np.isscalar(x):
            return np.vectorize(self.dipdf, otypes=[np.float64])(x)
        return self._diff(self.ipdf, x, eps)

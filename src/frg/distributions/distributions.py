"""
Distributions
-------------

Distributions used in experiments and simulations. The module provides both empirical and theoretical distributions to reproduce specific spectra of momenta.

Authors
-------

- Riccardo Finotello <riccardo.finotello@cea.fr>
- Parham Radpay <parhamradpay@gmail.com>

Maintainers
-----------

- Riccardo Finotello
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
from warnings import warn

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.stats import gaussian_kde
from skimage.transform import resize

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from jaxtyping import Float, Integer
    from numpy.random import Generator
    from yacs.config import CfgNode

    # Dummy variables for jaxtyping to prevent Ruff F821 errors.
    # Defined strictly within TYPE_CHECKING so they cannot exist at runtime,
    # ensuring they never overwrite or conflict with actual code variables.
    n_samples: int = 1000
    n_vars: int = 500
    n_samples_sig: int = 1000
    n_vars_sig: int = 500
    n_steps: int = 100
    n_spikes: int = 10


class Distribution:
    """An abstract base class for distributions."""

    def __init__(self):
        # Parameters (set to placeholder values)
        #
        # - m^2 : inverse of the largest eigenvalue (mass)
        # - l_- : lowest eigenvalue
        # - l_+ : largest eigenvalue
        self.m2: float = np.inf
        self.lminus: float = -np.inf
        self.lplus: float = np.inf

    @overload
    def pdf(self, x: float) -> float: ...

    @overload
    def pdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def pdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the value of the *Probability Density Function* (PDF) of a random variable :math:`X` at given value(s) :math:`x`.
        """
        raise NotImplementedError(
            "All sublasses of Distribution must implement a custom `pdf` method!"
        )

    @overload
    def cdf(self, x: float) -> float: ...

    @overload
    def cdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def cdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the value of the *Cumulative Distribution Function* (CDF) of a random variable :math:`X` at given value(s) :math:`x`.
        """
        raise NotImplementedError(
            "All sublasses of Distribution must implement a custom `cdf` method!"
        )

    @overload
    def dpdf(self, x: float, eps: float = 1.0e-6) -> float: ...

    @overload
    def dpdf(
        self, x: Float[np.ndarray, "n_samples"], eps: float = 1.0e-6
    ) -> Float[np.ndarray, "n_samples"]: ...

    def dpdf(
        self, x: float | Float[np.ndarray, "n_samples"], eps: float = 1.0e-6
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the derivative of the PDF of a random variable :math:`X` at given value(s) :math:`x`. The derivative is computed using the finite difference method:

        .. math::

            f'(x) \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2 \\epsilon}

        Parameters
        ----------
        x : float or ndarray of floats
            The point(s) at which to compute the derivative of the PDF (shape: scalar or array of shape :math:`n`).
        eps : float
            The incremental ratio :math:`\\epsilon`, by default :math:`10^{-9}`.

        Returns
        -------
        float or ndarray of floats
            The value(s) of the derivative of the PDF (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.dpdf, otypes=[np.float64])(x)
        return self._diff(self.pdf, x, eps)

    @overload
    def ipdf(self, x: float) -> float: ...

    @overload
    def ipdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...
    def ipdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the PDF of the inverse of the random variable :math:`X` (i.e. :math:`Y = 1 / X`) at given value(s) :math:`x`.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the PDF of the inverse variable (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the PDF of the inverse variable at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        # If x is not a scalar, then vectorize the function
        if not isinstance(x, float):
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

    @overload
    def lnipdf(self, x: float) -> float: ...

    @overload
    def lnipdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...
    def lnipdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the natural log-PDF of the inverse of the random variable :math:`X` (i.e. :math:`Y = 1 / X`) at given value(s) :math:`x`.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the log-PDF of the inverse variable (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the log-PDF of the inverse variable at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.lnipdf, otypes=[np.float64])(x)
        return float(np.log(self.ipdf(x)))

    @overload
    def icdf(self, x: float) -> float: ...

    @overload
    def icdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def icdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the CDF of the inverse of the random variable :math:`X` (i.e. :math:`Y = 1 / X`) at given value(s) :math:`x`.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the CDF of the inverse variable (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the CDF of the inverse variable at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.icdf, otypes=[np.float64])(x)
        if x <= 0.0:
            return 0.0
        return quad(self.ipdf, 0.0, x)[0]

    @overload
    def dipdf(self, x: float) -> float: ...

    @overload
    def dipdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def dipdf(
        self, x: float | Float[np.ndarray, "n_samples"], eps: float = 1.0e-6
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the derivative of the PDF of the inverse of the random variable :math:`X` (i.e. :math:`Y = 1 / X`) at given value(s) :math:`x`. The derivative is computed using a central difference method:

        .. math::

            f'(x) \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2 \\epsilon}

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the derivative of the inverse PDF (shape: scalar or array of shape :math:`n`).
        eps : float
            The incremental ratio :math:`\\epsilon`, by default :math:`10^{-9}`.

        Returns
        -------
        float or ndarray of floats
            The value(s) of the derivative of the inverse PDF at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.dipdf, otypes=[np.float64])(x)
        return self._diff(self.ipdf, x, eps)

    @overload
    def dlnipdf(self, x: float) -> float: ...

    @overload
    def dlnipdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def dlnipdf(
        self, x: float | Float[np.ndarray, "n_samples"], eps: float = 1.0e-6
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the derivative of the natural log-PDF of the inverse of the random variable :math:`X` (i.e. :math:`Y = 1 / X`) at given value(s) :math:`x`. The derivative is computed using a central difference method:

        .. math::

            f'(x) \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2 \\epsilon}

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the derivative of the natural log-inverse PDF (shape: scalar or array of shape :math:`n`).
        eps : float
            The incremental ratio :math:`\\epsilon`, by default :math:`10^{-9}`.

        Returns
        -------
        float or ndarray of floats
            The value(s) of the derivative of the natural log-inverse PDF at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.dlnipdf, otypes=[np.float64])(x)
        return self._diff(self.lnipdf, x, eps)

    def _diff(
        self, func: Callable[[float], float], x: float, eps: float = 1.0e-6
    ) -> float:
        """
        Compute a derivative of a function using a central difference method:

        .. math::

            f'(x) \\approx \\frac{f(x + \\epsilon) - f(x - \\epsilon)}{2 \\epsilon}

        Parameters
        ----------
        func: Callable
            The function to evaluate.
        x : float
            The value at which to evaluate the derivative.
        eps : float
            The step to compute the incremental ratio, by default :math:`10^{-9}`.

        Returns
        -------
        float
            The value of the derivative of the PDF at the given value.
        """
        num: float = func(x + eps) - func(x - eps)
        den: float = 2.0 * eps
        return num / den if den != 0.0 else 0.0

    @overload
    def canonical_dimensions(self, x: float) -> Float[np.ndarray, "4"]: ...

    @overload
    def canonical_dimensions(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples, 4"]: ...

    def canonical_dimensions(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "4"] | Float[np.ndarray, "n_samples, 4"]:
        """
        Compute the canonical dimensions of the distribution using the following formulae:

        .. math::

            \\text{dim}_{u_2}(k^2) = \\frac{\\int^{k^2}_0 \\mathrm{d}p^2 \\rho(p^2)}{k^2 \\rho(k^2)}

        .. math::

            \\text{dim}_{u_4}(k^2) = \\text{dim}_{u_2}(k^2) \\left( 3 + k^2 \\frac{\\partial \\ln \\rho(k^2)}{\\partial k^2} - 2 \\right)

        .. math::

            \\text{dim}_{u_6}(k^2) = -\\text{dim}_{u_2}(k^2) + 2 \\cdot \\text{dim}_{u_4}(k^2)

        .. math::

            \\text{dim}_{\\chi}(k^2)= 1 - \\text{dim}_{u_2}(k^2)

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the canonical dimensions (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        ndarray of floats
            An array containing, in columns (shape: :math:`4`, if input is scalar, or :math:`n \\times 4`, if input is array):

            - ``dim_u2``: the mass term
            - ``dim_u4``: the quartic interaction
            - ``dim_u6``: the sextic interaction
            - ``dim_chi``: the dimension of the field
        """
        if not isinstance(x, float):
            return np.vectorize(
                self.canonical_dimensions,
                otypes=[np.float64],
                signature="()->(n)",
            )(x)

        # Ignore the warning of zero value division and compute the dimensions
        with np.errstate(divide="ignore", invalid="ignore"):
            dimu2: float = self.icdf(x) / self.ipdf(x) / x
            dimu4: float = dimu2 * (3.0 + x * self.dlnipdf(x)) - 2.0
            dimu6: float = -dimu2 + 2.0 * dimu4
            dimchi: float = 1.0 - dimu2
        return np.array([dimu2, dimu4, dimu6, dimchi])

    def _frg_equations_single(
        self,
        x: float,
        u2: float,
        u4: float,
        u6: float,
        dx: float = 1.0e-9,
        return_rhs: bool = False,
    ) -> Float[np.ndarray, "3"]:
        """
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
            The step size, by default ``1.0e-9``.
        return_rhs : bool
            Whether to return the right-hand side of the FRG equations instead of updating the couplings, by default False.

        Returns
        -------
        ndarray of floats
            An array containing, in columns (shape: :math:`3`):

            - ``u2``: the running of the coupling :math:`u_2`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the prefactor
        pref: float = self.ipdf(x) / self.icdf(x)

        # Canonical dimensions
        dimu2: float
        dimu4: float
        dimu6: float
        dimu2, dimu4, dimu6, _ = self.canonical_dimensions(x).T

        # Right Hand Sides
        u2_rhs: float = -dimu2 * u2 - 2.0 * u4 / (1.0 + u2) ** 2
        u4_rhs: float = (
            -dimu4 * u4
            - 2.0 * u6 / (1.0 + u2) ** 2
            + 12.0 * u4**2 / (1.0 + u2) ** 3
        )
        u6_rhs: float = (
            -dimu6 * u6
            + 60.0 * u4 * u6 / (1.0 + u2) ** 3
            - 108.0 * u6**3 / (1.0 + u2) ** 4
        )

        # Return the RHS if requested by the solver
        if return_rhs:
            return pref * np.array([u2_rhs, u4_rhs, u6_rhs])

        # Equations (the - sign is due to the flow towards the IR)
        u2 -= dx * pref * u2_rhs
        u4 -= dx * pref * u4_rhs
        u6 -= dx * pref * u6_rhs

        return np.array([u2, u4, u6])

    def frg_equations(
        self,
        x: float,
        u2_init: float,
        u4_init: float,
        u6_init: float,
        dx: float = 1.0e-9,
        x_ir: float = 0.0,
        use_naive_euler: bool = True,
    ) -> Float[np.ndarray, "n_steps, 4"]:
        """
        Given initial conditions of the couplings :math:`u_2`, :math:`u_4`, and :math:`u_6` at a given momentum scale :math:`k^2`, the function returns the values of the couplings at the scale :math:`k^2 - \\Delta_k`. That is, the function returns the running of the couplings from the UV region towards the IR (:math:`k^2 \\to 0`):

        .. math::

            \\dot{u}_2 = - \\text{dim}_{u_2} u_2 - 2 \\frac{u_4}{(1 + u_2)^2}

        .. math::

            \\dot{u}_4 = - \\text{dim}_{u_4} u_4 - 2 \\frac{u_6}{(1 + u_2)^2} + 12 \\frac{u_4^2}{(1 + u_2)^3}

        .. math::

            \\dot{u}_6 = - \\text{dim}_{u_6} u_6 + 60 \\frac{u_4 u_6}{(1 + u_2)^3} - 108 \\frac{u_6^3}{(1 + u_2)^4}

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
            The step size, by default ``1.0e-9``.
        x_ir : float
            The definition of IR scale (i.e. the scale :math:`k^2` to be considered low-energy), by default 0.0. It can be overridden to match physical assumptions (e.g. histogram binning, etc.).
        use_naive_euler : bool
            Whether to use the naive Euler method for integration, by default True. If False, a more sophisticated integration method will be used.

        Returns
        -------
        ndarray of floats
            An array containing, in columns (shape: :math:`S \\times 4` where :math:`S` is the number of steps):

            - ``k2``: the energy scale corresponding to the couplings
            - ``u2``: the running of the coupling :math:`u_2`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`

            The number of computed steps is:

            .. math::

                S = \\lfloor \\frac{x - x_{\\text{IR}}}{\\Delta x} \\rfloor
        """
        # Compute the energy scale
        k2: float
        u2: float
        u4: float
        u6: float
        k2, u2, u4, u6 = x, u2_init, u4_init, u6_init
        if use_naive_euler:
            results = []
            while k2 >= max(dx, x_ir):
                u2, u4, u6 = self._frg_equations_single(
                    x=k2,
                    u2=u2,
                    u4=u4,
                    u6=u6,
                    dx=dx,
                    return_rhs=not use_naive_euler,
                )
                results.append([k2 - dx, u2, u4, u6])
                k2 -= dx
        else:
            # Define a closure with the signature needed by the solver
            def flow_derivative(
                t: float, y: Float[np.ndarray, "3"]
            ) -> Float[np.ndarray, "3"]:
                u2: float
                u4: float
                u6: float
                u2, u4, u6 = y
                return self._frg_equations_single(
                    x=t,
                    u2=u2,
                    u4=u4,
                    u6=u6,
                    dx=dx,
                    return_rhs=True,
                )

            sol = solve_ivp(
                fun=flow_derivative,
                t_span=(x, x_ir),  # UV to IR
                y0=[u2, u4, u6],
                method="Radau",
                dense_output=True,
            )
            k2_list: np.ndarray = sol.t
            u2_list: np.ndarray
            u4_list: np.ndarray
            u6_list: np.ndarray
            u2_list, u4_list, u6_list = sol.y
            results = np.column_stack((k2_list, u2_list, u4_list, u6_list))
        return np.array(results)

    def _frg_equations_lpa_single(
        self,
        x: float,
        kappa: float,
        u4: float,
        u6: float,
        dx: float = 1.0e-9,
        return_rhs: bool = False,
    ) -> Float[np.ndarray, "3"]:
        """
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
            The step size, by default ``1.0e-9``.
        return_rhs : bool
            Whether to return the right-hand side of the FRG equations instead of updating the couplings, by default False.

        Returns
        -------
        ndarray of floats
            An array containing, in columns (shape: :math:`3`):

            - ``kappa``: the running of the position of the zero :math:`kappa`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`
        """
        # Compute the prefactor
        pref: float = self.ipdf(x) / self.icdf(x)

        # Canonical dimensions
        dimu4: float
        dimu6: float
        dimchi: float
        _, dimu4, dimu6, dimchi = self.canonical_dimensions(x).T

        # Right Hand Sides
        kappa_rhs: float = (
            -dimchi * kappa
            + 2.0
            * (3.0 + 2.0 * kappa * u6 / u4)
            / (1.0 + 2.0 * kappa * u4) ** 2
        )
        u4_rhs: float = (
            -dimu4 * u4
            + dimchi * kappa * u6
            - 10.0 * u6 / (1.0 + 2.0 * kappa * u4) ** 2
            + 4.0
            * (3.0 * u4 + 2.0 * kappa * u6) ** 2
            / (1.0 + 2.0 * kappa * u4) ** 3
        )
        u6_rhs: float = (
            -dimu6 * u6
            - 12.0
            * (3.0 * u4 + 2.0 * kappa * u6) ** 3
            / (1.0 + 2.0 * kappa * u4) ** 4
            + 40.0
            * u6
            * (3.0 * u4 + 2.0 * kappa * u6)
            / (1.0 + 2.0 * kappa * u4)
        )

        # Return the continuous physical derivative for SciPy
        if return_rhs:
            return pref * np.array([kappa_rhs, u4_rhs, u6_rhs])

        # Equations (the - sign indicates the direction of the flow to IR)
        kappa -= dx * pref * kappa_rhs
        u4 -= dx * pref * u4_rhs
        u6 -= dx * pref * u6_rhs

        return np.array([kappa, u4, u6])

    def frg_equations_lpa(
        self,
        x: float,
        kappa_init: float,
        u4_init: float,
        u6_init: float,
        dx: float = 1.0e-9,
        x_ir: float = 0.0,
        use_naive_euler: bool = True,
    ) -> Float[np.ndarray, "n_steps, 4"]:
        """
        Given initial conditions of the zero of the potential :math:`\\kappa` and the couplings :math:`u_2`, and :math:`u_4` at a given momentum scale :math:`k^2`, the function returns the values of the couplings at the scale :math:`k^2 - \\Delta_k` in **Local Potential Approximation** (LPA). That is, the function returns the running of the couplings from the UV region towards the IR (:math:`k^2 \\to 0`):

        .. math::

            \\dot{\\kappa} = - \\text{dim}_{\\chi} \\kappa + 2 \\frac{3 + 2 \\kappa \\frac{u_6}{u_4}}{(1 + 2 \\kappa u_4)^2}

        .. math::

            \\dot{u}_4 = - \\text{dim}_{u_4} u_4 + \\text{dim}_{\\chi} \\kappa u_6 - 10 \\frac{u_6}{(1 + 2 \\kappa u_4)^2} + 4 \\frac{(3 u_4 + 2 \\kappa u_6)^2}{(1 + 2 \\kappa u_4)^3}

        .. math::

            \\dot{u}_6 = - \\text{dim}_{u_6} u_6 - 12 \\frac{(3 u_4 + 2 \\kappa u_6)^3}{(1 + 2 \\kappa u_4)^4} + 40 \\frac{u_6 (3 u_4 + 2 \\kappa u_6)}{(1 + 2 \\kappa u_4)^2}

        Parameters
        ----------
        x : float
            The momentum scale :math:`k^2` at which to start computing the FRG equations. This represents an energy scale in the UV region. The FRG computation ends in the IR region.
        kappa_init : float
            The initial value of the position of the zero of the potential :math:`\\kappa` at energy scale :math:`x`.
        u4_init : float
            The initial value of the coupling :math:`u_4` at energy scale :math:`x`.
        u6_init : float
            The initial value of the coupling :math:`u_6` at energy scale :math:`x`.
        dx : float
            The step size, by default 1.0e-9.
        x_ir : float
            The definition of IR scale (i.e. the scale :math:`k^2` to be considered low-energy), by default 0.0. It can be overridden to match physical assumptions (e.g. histogram binning, etc.).
        use_naive_euler : bool
            Whether to use the naive Euler method for integration, by default True. If False, a more sophisticated integration method will be used.

        Returns
        -------
        ndarray of floats
            An array containing, in columns (shape: :math:`S \\times 4` where :math:`S` is the number of steps):

            - ``k2``: the energy scale corresponding to the couplings
            - ``kappa``: the running of the zero of the potential :math:`kappa`
            - ``u4``: the running of the coupling :math:`u_4`
            - ``u6``: the running of the coupling :math:`u_6`

            The number of computed steps is:

            .. math::

                S = \\lfloor \\frac{x - x_{\\text{IR}}}{\\Delta x} \\rfloor
        """
        # Compute the energy scale
        k2: float
        kappa: float
        u4: float
        u6: float
        k2, kappa, u4, u6 = x, kappa_init, u4_init, u6_init

        if use_naive_euler:
            results = []
            while k2 >= max(dx, x_ir):
                kappa, u4, u6 = self._frg_equations_lpa_single(
                    x=k2,
                    kappa=kappa,
                    u4=u4,
                    u6=u6,
                    dx=dx,
                    return_rhs=not use_naive_euler,
                )
                results.append([k2 - dx, kappa, u4, u6])
                k2 -= dx
        else:
            # Define a closure with the signature needed by the SciPy solver
            def flow_derivative(
                t: float, y: Float[np.ndarray, "3"]
            ) -> Float[np.ndarray, "3"]:
                kappa_val: float
                u4_val: float
                u6_val: float
                kappa_val, u4_val, u6_val = y
                return self._frg_equations_lpa_single(
                    x=t,
                    kappa=kappa_val,
                    u4=u4_val,
                    u6=u6_val,
                    dx=dx,
                    return_rhs=True,
                )

            sol = solve_ivp(
                fun=flow_derivative,
                t_span=(x, x_ir),  # UV to IR
                y0=[kappa, u4, u6],
                method="Radau",
                dense_output=True,
            )
            k2_list: np.ndarray = sol.t
            kappa_list: np.ndarray
            u4_list: np.ndarray
            u6_list: np.ndarray
            kappa_list, u4_list, u6_list = sol.y
            results = np.column_stack((k2_list, kappa_list, u4_list, u6_list))

        return np.array(results)


class EmpiricalDistribution(Distribution):
    """The empirical distribution uses a **Kernel Density Estimation** (KDE) to fit the spectrum of the theory and analyse it."""

    def __init__(
        self,
        n_samples: int,
        var: float = 1.0,
        ratio: float = 0.5,
        is_poisson: bool = False,
        poisson_data: bool = False,
        poisson_lam: float = 10.0,
        poisson_centering: Literal[
            "centered", "non-centered", "mirrored"
        ] = "centered",
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        n_samples : int
            The number of samples in the distribution.
        var : float
            The variance of the distribution.
        ratio : float
            The ratio between the number of variables and the dimension of the sample/population.

            .. note::

                Let :math:`X \\in \\mathbb{R}^{n \\times p}` be a real matrix where samples are **row** vectors, then the ``ratio`` parameter is :math:`\\frac{p}{n}`.

        is_poisson : bool
            Add Poisson noise (e.g. photon counting), by default ``False``.
        poisson_data : bool
            Whether to add Poisson noise directly to the data (if ``True``) or to the covariance matrix (if ``False``), by default ``False``.
        poisson_lam : float
            The rate parameter for the Poisson distribution, by default ``10.0``.
        poisson_centering : Literal["centered", "non-centered", "mirrored"]
            The centering method for Poisson noise, by default ``"centered"``. Must be one of:
            - "centered": Center the Poisson noise by removing the expected value from the generated sample.
            - "non-centered": Do not center the Poisson noise.
            - "mirrored": Mirror the Poisson noise by generating two samples and subtracting them (Skellam).
        seed : int
            The random seed, by default ``42``.

        Raises
        ------
        ValueError
            If the size of the sample is less than 2
        ValueError
            If variance is not strictly positive
        ValueError
            If ratio is not strictly positive
        ValueError
            If the Poisson rate parameter is negative
        ValueError
            If poisson_centering is not one of "centered", "non-centered", or "mirrored".

        Warns
        -----
        UserWarning
            If the size of the sample is less than 1000 (sample too small)
        """
        super().__init__()
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
        self.n_samples: int = int(n_samples)
        if var <= 0.0:
            raise ValueError(
                "The variance must be strictly positive but got %f <= 0" % var
            )
        self.var: float = float(var)
        if ratio <= 0.0:
            raise ValueError(
                "Ratio must be strictly positive but got %f <= 0" % ratio
            )
        self.ratio: float = float(ratio)
        self.n_vars: int = int(n_samples * ratio)  # no. of vars / cols
        self.is_poisson: bool = bool(is_poisson)
        self.poisson_data: bool = bool(poisson_data)
        if poisson_lam < 0.0:
            raise ValueError(
                "The rate parameter for the Poisson distribution must be non-negative but got %f < 0"
                % poisson_lam
            )
        self.poisson_lam: float = float(poisson_lam)
        if poisson_centering not in ["centered", "non-centered", "mirrored"]:
            raise ValueError(
                "poisson_centering must be one of 'centered', 'non-centered', or 'mirrored' but got %s"
                % poisson_centering
            )
        self.poisson_centering: Literal[
            "centered", "non-centered", "mirrored"
        ] = poisson_centering
        self.seed: int = int(seed)
        self._fitted: bool = False
        self._iscov: bool = False  # if generated directly from covariance

        # Generate the background distribution
        gen: Generator = np.random.default_rng(self.seed)
        self._noise: Float[np.ndarray, "n_samples, n_vars"] = gen.normal(
            loc=0.0,
            scale=np.sqrt(self.var),
            size=(self.n_samples, self.n_vars),
        )

        # Handle Poisson noise
        #
        # Case 1: Poisson noise is added directly to the data
        #         ---> The noise matrix has size n x p
        if self.is_poisson and (self.poisson_data):
            if self.poisson_centering == "non-centered":
                self._noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                )
                self.var: float = self.poisson_lam
            elif self.poisson_centering == "centered":
                self._noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                    - self.poisson_lam
                )
                self.var: float = self.poisson_lam
            elif self.poisson_centering == "mirrored":
                self._noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                    - gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                )
                self.var: float = 2.0 * self.poisson_lam
        #
        # Case 2: Poisson noise is added to the covariance matrix
        #         ---> The noise matrix has size p x p
        if self.is_poisson and (not self.poisson_data):
            if self.poisson_centering == "non-centered":
                base_noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                )
            elif self.poisson_centering == "centered":
                base_noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                    - self.poisson_lam
                )
            elif self.poisson_centering == "mirrored":
                base_noise: Float[np.ndarray, "n_samples, n_vars"] = (
                    gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                    - gen.poisson(
                        lam=self.poisson_lam,
                        size=(self.n_samples, self.n_vars),
                    )
                )

            # Symmetric noise matrix (positive semi-definite)
            self._cov_noise: Float[np.ndarray, "n_vars, n_vars"] = (
                base_noise.T @ base_noise
            ) / max(1.0, float(self.n_samples - 1))

        # Prepare data
        self.data: Float[np.ndarray, "n_samples, n_vars"] = self._noise
        self.eigenvectors_: Float[np.ndarray, "n_samples, n_vars"] | None = None
        self.eigenvalues_: Float[np.ndarray, "n_vars"] | None = None

    @property
    def data(self) -> Float[np.ndarray, "n_samples, n_vars"]:
        """
        Return the data matrix.

        Returns
        -------
        ndarray of floats
            The data matrix (shape: :math:`n \\times p`).
        """
        return self._data

    @data.setter
    def data(self, value: Float[np.ndarray, "n_samples, n_vars"]) -> None:
        """
        Set the data matrix.

        Parameters
        ----------
        value : ndarray of floats
            The data matrix (shape: :math:`n \\times p`).

        Raises
        ------
        ValueError
            If the data matrix is not 2-dimensional.
        """
        if value.ndim != 2:
            raise ValueError(
                "The data matrix must be 2-dimensional but got %d-dimensional"
                % value.ndim
            )
        self._data: Float[np.ndarray, "n_samples, n_vars"] = value

    @classmethod
    def from_config(cls, cfg: CfgNode) -> EmpiricalDistribution:
        """
        Create an instance of the class from a configuration.

        Parameters
        ----------
        cfg : CfgNode
            The configuration.

        Returns
        -------
        EmpiricalDistribution
            An instance of the class.
        """
        return cls(
            n_samples=cfg.DIST.NUM_SAMPLES,
            var=cfg.DIST.VAR,
            ratio=cfg.DIST.RATIO,
            seed=cfg.DIST.SEED,
            is_poisson=cfg.DIST.IS_POIS,
            poisson_data=cfg.DIST.POIS_DATA,
            poisson_lam=cfg.DIST.POIS_LAM,
            poisson_centering=cfg.DIST.POIS_MODE,
        )

    @classmethod
    def from_covariance(
        cls, cov: Float[np.ndarray, "n_samples, n_vars"], cfg: CfgNode
    ) -> EmpiricalDistribution:
        """
        Create an instance of the class from a covariance matrix.

        Parameters
        ----------
        cov : ndarray of floats
            The covariance matrix (shape: :math:`n \\times p`).
        cfg : CfgNode
            The configuration

        Returns
        -------
        EmpiricalDistribution
            An instance of the class.
        """
        instance: EmpiricalDistribution = cls.from_config(cfg)
        instance.data: Float[np.ndarray, "n_samples, n_vars"] = cov
        instance._iscov = True
        return instance

    def add_signal(
        self,
        X: Float[np.ndarray, "n_samples_sig, n_vars_sig"],
        snr: float = 0.0,
    ) -> EmpiricalDistribution:
        """
        Add a signal to the distribution.

        Parameters
        ----------
        X : ndarray of floats
            The signal to add (shape: :math:`n_S \\times p_S`). If :math:`n_S \\neq n` or :math:`p_S \\neq p`, the signal matrix will be resized to match the background.
        snr : float
            The signal-to-noise ratio, by default ``0.0``. If negative, then only signal is kept and noise is discarded.

        Returns
        -------
        EmpiricalDistribution
            A new instance of the class with the signal added.

        Raises
        ------
        ValueError
            If the signal matrix is not 2-dimensional.

        Warns
        -----
        UserWarning
            If the distribution was created from a covariance matrix, adding a signal may lead to unexpected results.
        """
        if self._iscov:
            warn(
                "You are adding a signal to a distribution created from a covariance matrix. This may lead to unexpected results."
            )
        if X.ndim != 2:
            raise ValueError(
                "The signal matrix must be 2-dimensional but got %d-dimensional"
                % X.ndim
            )

        # Add the signal to the background
        n: int
        p: int
        n, p = self.data.shape
        nS: int
        pS: int
        nS, pS = X.shape
        signal: Float[np.ndarray, "n_samples_sig, n_vars_sig"] = (
            resize(X, output_shape=self.data.shape[:2])
            if ((n != nS) or (p != pS))
            else X
        )
        if snr >= 0.0:
            self.data += snr * signal
        else:
            self.data: Float[np.ndarray, "n_samples, n_vars"] = signal

        return self

    def _cov_eigh(
        self, X: Float[np.ndarray, "n_samples, n_vars"], is_cov: bool = False
    ) -> np.ndarray:
        """
        Get the eigenvalues of the covariance matrix of the data (from the singular values of the data).

        Parameters
        ----------
        X : ndarray of floats
            The data matrix (shape: :math:`n \\times p`).
        is_cov : bool
            Data are a covariance matrix, by default `False`.

        Returns
        -------
        np.ndarray
            The eigenvalues of the distribution.
        """
        if is_cov:
            cov: Float[np.ndarray, "n_vars, n_vars"] = X.copy()
        else:
            cov: Float[np.ndarray, "n_vars, n_vars"] = (X.T @ X) / max(
                1.0, float(self.n_samples - 1.0)
            )
        if hasattr(self, "_cov_noise"):
            cov += self._cov_noise

        # Compute the eigendecomposition
        evl: Float[np.ndarray, "n_vars"]
        evc: Float[np.ndarray, "n_vars, n_vars"]
        evl, evc = np.linalg.eigh(cov)
        self.eigenvalues_: Float[np.ndarray, "n_vars"] = evl
        self.eigenvectors_: Float[np.ndarray, "n_vars, n_vars"] = evc
        return evl

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Compute the eigenvalues of the distribution.

        .. note::

            This is the complete list of eigenvalues (bulk + spikes).
            You can access the filtered distribution **without** the spikes using the ``self.eigenvalues`` attribute, available after calling the ``self.fit(...)`` method.

        Returns
        -------
        np.ndarray
            The eigenvalues of the distribution, sorted in ascending order.
        """
        evl: Float[np.ndarray, "n_vars"] = self._cov_eigh(
            self.data, is_cov=self._iscov
        )
        assert self.eigenvalues_ is not None, "Error: Eigenvalues not computed!"
        assert self.eigenvectors_ is not None, (
            "Error: Eigenvectors not computed!"
        )

        # Sort the eigenvalues
        idx: Integer[np.ndarray, "n_vars"] = np.argsort(evl)
        self.eigenvalues_ = evl[idx]
        self.eigenvectors_ = self.eigenvectors_[:, idx]
        return evl[idx]

    def find_spikes(
        self, eigenvalues: Float[np.ndarray, "n_vars"], pow: float = 0.5
    ) -> int:
        """
        Find the spikes in the eigenvalues.

        The search for spikes is done by measuring the distance between eigenvalues from the IR to the UV. When eigenvalues start to form a continuous bulk (i.e. the difference between consecutive eigenvalues is smaller than a threshold), they are considered part of the bulk.

        Parameters
        ----------
        eigenvalues : ndarray of floats
            The eigenvalues of the distribution (shape: :math:`p`).
        pow : float
            Continuity criterion of the eigenvalues, by default ``0.5``.

        Returns
        -------
        int
            The index of the first spike (which equals the number of eigenvalues in the bulk).
        """
        dx: float = len(eigenvalues) ** (-pow)

        # Find the first gap from the IR (left) that is larger than dx
        diff: Float[np.ndarray, "n_vars - 1"] = np.diff(eigenvalues)
        large_gaps: np.ndarray = np.where(diff > dx)[0]

        if len(large_gaps) == 0:
            # All gaps are small, the entire spectrum is considered
            # the continuous bulk
            return len(eigenvalues)

        if len(large_gaps) == len(diff) or np.median(diff) > dx:
            # All gaps are uniformly large (e.g. scaled variance)
            # Only the most UV eigenvalue is considered a spike
            return 1

        # The bulk ends right before the first large gap
        return int(large_gaps[0]) + 1

    def _find_pow(self, pow: float, snr: float) -> float:
        """
        Interpolate a power factor as a function of the SNR to ensure to eliminate all spikes.

        Parameters
        ----------
        pow : float
            The power factor.
        snr : float
            The actual signal-to-noise ratio.

        Returns
        -------
        float
            The spike-finding factor.
        """
        coeff: float = (pow - 0.5) / 5.0
        return coeff * snr + 0.5

    def fit(
        self,
        X: Float[np.ndarray, "n_samples, n_vars"] | None = None,
        snr: float = 0.0,
        fac: float = 0.3,
    ) -> EmpiricalDistribution:
        """
        Add the signal (if provided) and compute the eigenvalue distribution.

        Parameters
        ----------
        X : ndarray of floats, optional
            The signal to add (shape: :math:`n \\times p`).
        snr : float
            The signal-to-noise ratio.
        fac : float
            Bandwidth factor. By deault `0.3`.

        Returns
        -------
        EmpiricalDistribution
            A new instance of the class with the signal added and the eigenvalue distribution computed.
        """
        if X is not None:
            self.add_signal(X, snr=snr)

        # Remove the spikes from the eigenvalues and fit a KDE
        evls: Float[np.ndarray, "n_vars"] = self.eigenvalues
        spikes: int = self.find_spikes(
            eigenvalues=evls, pow=self._find_pow(pow=self.ratio, snr=snr)
        )

        assert self.eigenvalues_ is not None, "Error: Eigenvalues not computed!"
        assert self.eigenvectors_ is not None, (
            "Error: Eigenvectors not computed!"
        )
        self.eigenvalues_: Float[np.ndarray, "n_vars - n_spikes"] = evls[
            :spikes
        ]
        self.eigenvectors_: Float[np.ndarray, "n_vars, n_vars - n_spikes"] = (
            self.eigenvectors_[:, :spikes]
        )

        # Avoid KDE crash if there's only 1 eigenvalue left in the bulk
        if len(self.eigenvalues_) == 1:
            val: float = float(self.eigenvalues_[0])
            self.eigenvalues_ = np.array([val - 1.0e-5, val + 1.0e-5])

        self.kde = gaussian_kde(
            self.eigenvalues_,
            bw_method=lambda obj: np.power(obj.n, -1.0 / (obj.d + 4.0)) * fac,
        )

        # Compute the min and max points
        self.lminus: float = float(self.var * (1.0 - np.sqrt(self.ratio)) ** 2)
        self.lplus_mp: float = float(
            self.var * (1.0 + np.sqrt(self.ratio)) ** 2
        )
        self.lplus: float = float(max(self.eigenvalues_))
        self.m2: float = 1.0 / (self.lplus - self.lminus)
        self.m2_mp: float = 1.0 / (self.lplus_mp - self.lminus)
        self.norm: float = float(
            self.kde.integrate_box_1d(self.lminus, self.lplus)
        )

        self._fitted: bool = True
        return self

    @overload
    def pdf(self, x: float) -> float: ...

    @overload
    def pdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def pdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the value of the *Probability Density Function* (PDF) of the eigenvalue distribution at given value(s) :math:`x`.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the PDF (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the PDF at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling `pdf`! Please call `self.fit()` first."
            )
        if not isinstance(x, float):
            return np.vectorize(self.pdf, otypes=[np.float64])(x)

        if (x <= self.lminus) or (x >= self.lplus):
            return 0.0

        return float(self.kde.pdf(x)) / self.norm

    @overload
    def cdf(self, x: float) -> float: ...

    @overload
    def cdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def cdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the CDF of the eigenvalue distribution.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the CDF (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the CDF at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling `cdf`! Please call `self.fit()` first."
            )
        if not isinstance(x, float):
            return np.vectorize(self.cdf, otypes=[np.float64])(x)

        # Physical bounds
        if x <= self.lminus:
            return 0.0
        if x >= self.lplus:
            return 1.0

        val: float = float(self.kde.integrate_box_1d(self.lminus, x))
        return val / self.norm

    @overload
    def icdf(self, x: float) -> float: ...

    @overload
    def icdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def icdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the CDF of the distribution of the inverse of the eigenvalues.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the CDF of the momenta (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the CDF of the momenta (shape: scalar or array of shape :math:`n`).
        """
        if not self._fitted:
            raise ValueError(
                "The distribution must be fitted before calling ipdf! Please call ``self.fit()`` first."
            )
        if not isinstance(x, float):
            return np.vectorize(self.icdf, otypes=[np.float64])(x)

        # Let
        #     lambda' = lambda - lambda_-
        # and
        #     lambda' = 1 / (k^2 + m^2) <==> k^2 = 1 / lambda' - self.m2.
        #
        # We can then compute the CDF of the random variable K^2 by computing:
        #
        #    P(K^2 < k^2) = P(1 / lambda' < k^2 + m^2)
        #                 = P(lambda' > 1 / (k^2 + m^2))
        #                 = P(lambda > lambda_- + 1 / (k^2 + m^2))
        #                 = 1 - P(lambda < lambda_- + 1 / (k^2 + m^2))
        #                 = 1 - CDF(lambda_- + 1 / (k^2 + m^2))
        if x <= 0.0:
            return 0.0
        return 1.0 - self.cdf(self.lminus + 1.0 / (x + self.m2))


class MarchenkoPastur(Distribution):
    """
    The **Marchenko-Pastur** distribution represents the distribution of eigenvalues of large random covariance matrices.
    """

    def __init__(self, ratio: float, var: float):
        """
        Parameters
        ----------
        ratio : float
            The ratio between the number of variables and the dimension of the sample/population.

            .. note::

                Let :math:`X \\in \\mathbb{R}^{n \\times p}` be a real matrix where samples are **row** vectors, then the ``ratio`` parameter is :math:`\\frac{p}{n}`.

        var : float
            The variance of the distribution.

        Raises
        ------
        ValueError
            If ratio is not strictly positive
        ValueError
            If the lowest eigenvalue is higher than the highest one

        Warns
        -----
        UserWarning
            If ratio is higher than 1 because of numerical instabilities
        """
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
        if var < 0.0:
            raise ValueError(
                "Variance must be non-negative but got %f < 0" % var
            )

        # Store the parameters
        self.ratio: float = float(ratio)
        self.var: float = float(var)

        # Compute the highest and lowest eigenvalues
        self.lplus: float = self.var * (1.0 + np.sqrt(self.ratio)) ** 2
        self.lminus: float = self.var * (1.0 - np.sqrt(self.ratio)) ** 2
        self.m2: float = 1 / (self.lplus - self.lminus)
        if self.lminus > self.lplus:
            raise ValueError(
                "The smallest eigenvalue must be lower than the largest one, but got lminus = %f > %f = lplus!"
                % (self.lminus, self.lplus)
            )

    @overload
    def pdf(self, x: float) -> float: ...

    @overload
    def pdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def pdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the PDF of the Marchenko-Pastur distribution:

        .. math::

            \\mu_{\\text{MP}}(x) = \\frac{\\sqrt{(\\lambda_+ - x)(x - \\lambda_-)}}{2 \\pi \\sigma^2 q \\, x},

        where

        .. math::

            \\lambda_{\\pm} = \\sigma^2 (1 \\pm \\sqrt{q})^2,

        and

        .. math::

            q = \\frac{p}{n}.

        Parameters
        ----------
        x : float or ndarray of floats
            The value(s) at which to evaluate the PDF (shape: scalar or array of shape :math:`n`).

        Returns
        -------
        float or ndarray of floats
            The value(s) of the PDF at the given value(s) (shape: scalar or array of shape :math:`n`).
        """
        if not isinstance(x, float):
            return np.vectorize(self.pdf, otypes=[np.float64])(x)
        if (x <= self.lminus) or (x >= self.lplus):
            return 0.0

        num: float | Float[np.ndarray, "n_samples"] = np.sqrt(
            (self.lplus - x) * (x - self.lminus)
        )
        den: float | Float[np.ndarray, "n_samples"] = (
            2.0 * np.pi * self.var * self.ratio * x
        )
        return num / den if den != 0.0 else 0.0

    @overload
    def cdf(self, x: float) -> float: ...

    @overload
    def cdf(
        self, x: Float[np.ndarray, "n_samples"]
    ) -> Float[np.ndarray, "n_samples"]: ...

    def cdf(
        self, x: float | Float[np.ndarray, "n_samples"]
    ) -> float | Float[np.ndarray, "n_samples"]:
        """
        Compute the CDF of the Marchenko-Pastur distribution.

        Parameters
        ----------
        x : float | np.ndarray
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        float | np.ndarray
            The value(s) of the CDF at the given value(s).
        """
        if not isinstance(x, float):
            return np.vectorize(self.cdf, otypes=[np.float64])(x)

        if x <= self.lminus:
            return 0.0
        if x >= self.lplus:
            return 1.0
        return quad(lambda y: self.pdf(y), self.lminus, x)[0]

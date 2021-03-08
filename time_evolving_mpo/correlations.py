# Copyright 2021 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module for environment correlations.
"""

from typing import Callable, Dict, Optional, Text
from typing import Any as ArrayLike
import functools

import numpy as np
from scipy import integrate

from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import INTEGRATE_EPSREL, SUBDIV_LIMIT


# --- spectral density classes ------------------------------------------------

class BaseCorrelations(BaseAPIClass):
    """Base class for environment auto-correlations. """

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Auto-correlation function.

        .. math::

            C(\tau) = C(t, t-\tau) \
                    = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

        where :math:`\tau` is the time difference `tau` and :math:`F(t)` is the
        the environment part of the coupling operator in Heisenberg picture with
        respect to the environment Hamiltonian.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        raise NotImplementedError(
            "{} has no correlation implementation.".format(type(self).__name__))

    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{lower-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{t'-t_1}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        ``'lower-triangle'``, or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intevals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'lower-triangle'``, ``'lower-triangle'``}
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        raise NotImplementedError(
            "{} has no correlation_2d_integral implementation.".format(
                type(self).__name__))


    @property
    def max_correlation_time(self):
        """Maximal correlation time. """
        return self._max_correlation_time

    @max_correlation_time.setter
    def max_correlation_time(self, new_tau: Optional[float] = None):
        if new_tau is None:
            del self.max_correlation_time
        else:
            # check input: cutoff
            try:
                __new_tau = float(new_tau)
            except Exception as e:
                raise AssertionError( \
                    "Maximum correlation time must be a float.") from e
            if __new_tau < 0:
                raise ValueError(
                    "Maximum correlation time must be non-negative.")
            self._max_correlation_time = __new_tau

    @max_correlation_time.deleter
    def max_correlation_time(self):
        self._max_correlation_time = None


class CustomCorrelations(BaseCorrelations):
    r"""
    Encodes a custom auto-correlation function

    .. math::

        C(\tau) = C(t, t-\tau) = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

    with time difference `tau` :math:`\tau` and :math:`F(t)` is the
    the environment part of the coupling operator in Heisenberg picture with
    respect to the environment Hamiltonian. We assume that :math:`C(\tau) = 0`
    for all :math:`\tau > \tau_\mathrm{max}`.

    Parameters
    ----------
    correlation_function : callable
        The correlation function :math:`C`.
    max_correlation_time : float
        The maximal occuring correlation time :math:`\tau_\mathrm{max}`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    description_dict: dict
        An optional dictionary with descriptive data.
    """

    def __init__(
            self,
            correlation_function: Callable[[float], float],
            max_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Creates a CustomCorrelations object. """

        # check input: j_function
        try:
            __correlation_function = np.vectorize(correlation_function)
            complex(__correlation_function(1.0))
        except Exception as e:
            raise AssertionError("Correlation function must be vectorizable " \
                + "and must return float.") from e
        self.correlation_function = __correlation_function

        self.max_correlation_time = max_correlation_time

        super().__init__(name, description, description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append(f"  max_correlation_time = {self.max_correlation_time} \n")
        return "".join(ret)

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = None,
            subdiv_limit: Optional[int] = None) -> ArrayLike:
        r"""
        Auto-correlation function.

        .. math::

            C(\tau) = C(t, t-\tau) \
                    = \langle F(t) F(t-\tau) \rangle_\mathrm{env}

        with time difference `tau` :math:`\tau` and :math:`F(t)` is the
        the environment part of the coupling operator in Heisenberg picture with
        respect to the environment Hamiltonian.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tollerance (has no effect here).
        subdiv_limit : int
            Maximal number of interval subdivisions for numerical integration
            (has no effect here).

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        return self.correlation_function(tau)

    @functools.lru_cache(maxsize=2**10, typed=False)
    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{lower-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{t'-t_1}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        ``'lower-triangle'``, or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intevals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'lower-triangle'``, ``'lower-triangle'``}
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        c_real = lambda y, x: np.real(self.correlation(x-y))
        c_imag = lambda y, x: np.imag(self.correlation(x-y))

        if time_2 is None:
            time_2 = time_1 + delta
        else:
            assert shape == 'rectangle', \
                "paramter 'time_2' can only be used in conjunction with " \
                "'shape' = ``rectangle`` !"

        lower_boundary = {'square': lambda x: 0.0,
                          'upper-triangle': lambda x: 0.0,
                          'lower-triangle': lambda x: x-time_1,
                          'rectangle': lambda x: 0.0}
        upper_boundary = {'square': lambda x: delta,
                          'upper-triangle': lambda x: x-time_1,
                          'lower-triangle': lambda x: delta,
                          'rectangle': lambda x: delta,}

        int_real = integrate.dblquad(func=c_real,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        int_imag = integrate.dblquad(func=c_imag,
                                     a=time_1,
                                     b=time_2,
                                     gfun=lower_boundary[shape],
                                     hfun=upper_boundary[shape],
                                     epsrel=epsrel)[0]
        return int_real+1.0j*int_imag


# === CORRELATIONS FROM SPECTRAL DENSITIES ====================================


# --- the cutoffs -------------------------------------------------------------

def _hard_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)

def _exponential_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Exponential cutoff function."""
    return np.exp(-omega/omega_c)

def _gaussian_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Gaussian cutoff function."""
    return np.exp(-(omega/omega_c)**2)

# dictionary for the various cutoffs in the form:
#   'cutoff_name': cutoff_function
CUTOFF_DICT = {
    'hard':_hard_cutoff,
    'exponential':_exponential_cutoff,
    'gaussian':_gaussian_cutoff,
    }

# --- 2d integrals ------------------------------------------------------------

def _2d_square_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * 2 * np.cos(time_1*omega) * (1 - np.cos(delta*omega))

def _2d_square_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    integrand = 1.0/omega**2 * 2 * np.cos(time_1*omega) \
                * (1 - np.cos(delta*omega))
    return integrand / np.tanh(omega/(2*temperature))

def _2d_square_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    return -1.0/omega**2 * 2 * np.sin(time_1*omega) * (1 - np.cos(delta*omega))

def _2d_upper_triangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1+delta)) \
                           - omega * delta * np.sin(omega*time_1))

def _2d_upper_triangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1+delta)) \
                           - omega * delta * np.sin(omega*time_1)) \
                        / np.tanh(omega/(2*temperature))

def _2d_upper_triangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of upper triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin(omega*time_1) \
                            - np.sin(omega*(time_1+delta)) \
                            + omega * delta * np.cos(omega*time_1))

def _2d_lower_triangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1-delta)) \
                           + omega * delta * np.sin(omega*time_1))

def _2d_lower_triangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos(omega*time_1) \
                           - np.cos(omega*(time_1-delta)) \
                           + omega * delta * np.sin(omega*time_1)) \
                        / np.tanh(omega/(2*temperature))

def _2d_lower_triangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of lower triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin(omega*time_1)
                            - np.sin(omega*(time_1-delta))
                            - omega * delta * np.cos(omega*time_1))

def _2d_rectangle_integrand_real(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of rectangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (np.cos((time_2 - delta) * omega)
                           - np.cos((time_1 - delta) * omega)
                           - np.cos(time_2 * omega)
                           + np.cos(time_1 * omega))

def _2d_rectangle_integrand_real_t(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for real part of rectangle 2D time integral at finite
    temperature without J(omega). """
    integrand = 1.0/omega**2 * (np.cos((time_2 - delta) * omega)
                                - np.cos((time_1 - delta) * omega)
                                - np.cos(time_2 * omega)
                                + np.cos(time_1 * omega))
    return integrand / np.tanh(omega/(2*temperature))

def _2d_rectangle_integrand_imag(
        omega: ArrayLike,
        delta: float,
        time_1: float,
        time_2: Optional[float] = None,
        temperature: Optional[float] = None) -> ArrayLike:
    """Integrand for imaginary part of rectangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (np.sin((time_2 - delta) * omega)
                            - np.sin((time_1 - delta) * omega)
                            - np.sin(time_2 * omega)
                            + np.sin(time_1 * omega))

# dictionary for the various integrands for the 2d time integral
INTEGRAND_DICT = {
    'square': (_2d_square_integrand_real,
               _2d_square_integrand_real_t,
               _2d_square_integrand_imag),
    'upper-triangle': (_2d_upper_triangle_integrand_real,
                       _2d_upper_triangle_integrand_real_t,
                       _2d_upper_triangle_integrand_imag),
    'lower-triangle': (_2d_lower_triangle_integrand_real,
                       _2d_lower_triangle_integrand_real_t,
                       _2d_lower_triangle_integrand_imag),
    'rectangle': (_2d_rectangle_integrand_real,
                  _2d_rectangle_integrand_real_t,
                  _2d_rectangle_integrand_imag),
    }


# --- the spectral density classes --------------------------------------------

class CustomSD(BaseCorrelations):
    r"""
    Correlations corresponding to a custom spectral density. The resulting
    spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c) ,

    with `j_function` :math:`j`, `cutoff` :math:`\omega_c` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.


    Parameters
    ----------
    j_function : callable
        The spectral density :math:`j` without the cutoff.
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    max_correlation_time : float
        The maximal occuring correlation time :math:`\tau_\mathrm{max}`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    description_dict: dict
        An optional dictionary with descriptive data.
    """

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            cutoff_type: Optional[Text] = 'exponential',
            temperature: Optional[float] = 0.0,
            max_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a CustomFunctionSD (spectral density) object. """

        # check input: j_function
        try:
            __j_function = np.vectorize(j_function)
            float(__j_function(1.0))
        except Exception as e:
            raise AssertionError("Spectral density must be vectorizable " \
                + "and must return float.") from e
        self.j_function = __j_function

        # check input: cutoff
        try:
            __cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = __cutoff

        # check input: cutoff_type
        assert cutoff_type in CUTOFF_DICT, \
            "Cutoff type must be one of: {}".format(CUTOFF_DICT.keys())
        self.cutoff_type = cutoff_type

        # input check for temperature.
        try:
            __temperature = float(temperature)
        except Exception as e:
            raise AssertionError("Temperature must be a float.") from e
        if __temperature < 0.0:
            raise ValueError("Temperature must be >= 0.0 (but is {})".format(
                __temperature))
        self.temperature = __temperature

        self.max_correlation_time = max_correlation_time

        self._cutoff_function = \
            lambda omega: CUTOFF_DICT[self.cutoff_type](omega, self.cutoff)
        self._spectral_density = \
            lambda omega: self.j_function(omega) * self._cutoff_function(omega)

        super().__init__(name, description, description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  cutoff        = {} \n".format(self.cutoff))
        ret.append("  cutoff_type   = {} \n".format(self.cutoff_type))
        ret.append("  temperature   = {} \n".format(self.temperature))

        return "".join(ret)

    def spectral_density(self, omega: ArrayLike) -> ArrayLike:
        r"""
        The resulting spectral density (including the cutoff).

        Parameters
        ----------
        omega : ndarray
            The frequency :math:`\omega` for which we want to know the
            spectral density.

        Returns
        -------
        spectral_density : ndarray
            The resulting spectral density :math:`J(\omega)` at the frequency
            :math:`\omega`.
        """
        return self._spectral_density(omega)

    def correlation(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> ArrayLike:
        r"""
        Auto-correlation function associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            C(\tau) = \int_0^{\infty} J(\omega) \
                       \left[ \cos(\omega \tau) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * np.cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * np.cos(w*tau) \
                        / np.tanh(w/(2.0*self.temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) \
                                      * np.sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
        return re_int+1j*im_int

    @functools.lru_cache(maxsize=2**10, typed=False)
    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{lower-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{t'-t_1}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        ``'lower-triangle'``, or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intevals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'lower-triangle'``, ``'lower-triangle'``}
        epsrel : float
            Relative error tollerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        # real and imaginary part of the integrand
        if self.temperature == 0.0:
            re_integrand = lambda w: \
                    self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][0](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)
        else:
            re_integrand = lambda w: \
                    self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][1](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)
        im_integrand = lambda w: \
                    self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][2](w,
                                               delta,
                                               time_1,
                                               time_2,
                                               self.temperature)
        # real and imaginary part of the integral
        re_int = integrate.quad(re_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        im_int = integrate.quad(im_integrand,
                                a=0.0,
                                b=self.cutoff,
                                epsrel=epsrel,
                                limit=subdiv_limit)[0]
        if self.cutoff_type != "hard":
            re_int += integrate.quad(re_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]
            im_int += integrate.quad(im_integrand,
                                     a=self.cutoff,
                                     b=np.inf,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)[0]

        return re_int+1j*im_int


class PowerLawSD(CustomSD):
    r"""
    Correlations corresponding to the spectral density of the standard form

    .. math::

        J(\omega) = 2 \alpha \frac{\omega^\zeta}{\omega_c^{\zeta-1}} \
                    X(\omega,\omega_c)

    with `alpha` :math:`\alpha`, `zeta` :math:`\zeta` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.

    Parameters
    ----------
    alpha : float
        The coupling strenght :math:`\alpha`.
    zeta : float
        The exponent :math:`\zeta` (corresponds to the dimensionality of the
        environment). The environment is called *ohmic* if :math:`\zeta=1`,
        *superohmic* if :math:`\zeta>1` and *subohmic* if :math:`\zeta<1`
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    cutoff_type : str (default = ``'exponential'``)
        The cutoff type. Types are: {``'hard'``, ``'exponential'``,
        ``'gaussian'``}
    temperature: float
        The environment's temperature.
    max_correlation_time : float
        The maximal occuring correlation time :math:`\tau_\mathrm{max}`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    description_dict: dict
        An optional dictionary with descriptive data.
    """

    def __init__(
            self,
            alpha: float,
            zeta: float,
            cutoff: float,
            cutoff_type: Text = 'exponential',
            temperature: Optional[float] = 0.0,
            max_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a StandardSD (spectral density) object. """

        # check input: alpha
        try:
            __alpha = float(alpha)
        except Exception as e:
            raise AssertionError("Alpha must be a float.") from e
        self.alpha = __alpha

        # check input: zeta
        try:
            __zeta = float(zeta)
        except Exception as e:
            raise AssertionError("Nu must be a float.") from e
        self.zeta = __zeta

        # check input: cutoff
        try:
            __cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = __cutoff

        # use parent class for all the rest.
        j_function = lambda w: 2.0 * self.alpha * w**self.zeta \
                                    * self.cutoff**(1-zeta)

        self.max_correlation_time = max_correlation_time

        super().__init__(j_function,
                         cutoff=cutoff,
                         cutoff_type=cutoff_type,
                         temperature=temperature,
                         max_correlation_time=max_correlation_time,
                         name=name,
                         description=description,
                         description_dict=description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  alpha         = {} \n".format(self.alpha))
        ret.append("  zeta          = {} \n".format(self.zeta))

        return "".join(ret)

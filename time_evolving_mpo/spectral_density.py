# Copyright 2020 The TEMPO Collaboration
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
Module for spectral densities.
"""

from typing import Callable, Dict, Optional, Text
from typing import Any as ArrayLike

import numpy as np
from numpy import cos, sin, tanh, exp, vectorize

from time_evolving_mpo.integrate import integrate_semi_infinite
from time_evolving_mpo.base_api import BaseAPIClass

# --- the cutoffs -------------------------------------------------------------

def _hard_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)

def _exponential_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Exponential cutoff function."""
    return exp(-omega/omega_c)

def _gaussian_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Gaussian cutoff function."""
    return exp(-(omega/omega_c)**2)

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
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for real part of square 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * 2 * cos(time_1*omega) * (1 - cos(delta*omega))

def _2d_square_integrand_real_t(
        omega: ArrayLike,
        time_1: float,
        delta: float,
        temperature: float) -> ArrayLike:
    """Integrand for real part of square 2D time integral at finite
    temperature without J(omega). """
    integrand = 1.0/omega**2 * 2 * cos(time_1*omega) * (1 - cos(delta*omega))
    return integrand / tanh(omega/(2*temperature))

def _2d_square_integrand_imag(
        omega: ArrayLike,
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for imaginary part of square 2D time integral without
    J(omega). """
    return -1.0/omega**2 * 2 * sin(time_1*omega) * (1 - cos(delta*omega))

def _2d_upper_triangle_integrand_real(
        omega: ArrayLike,
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (cos(omega*time_1) \
                           - cos(omega*(time_1+delta)) \
                           - omega * delta * sin(omega*time_1))

def _2d_upper_triangle_integrand_real_t(
        omega: ArrayLike,
        time_1: float,
        delta: float,
        temperature: float) -> ArrayLike:
    """Integrand for real part of upper triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (cos(omega*time_1) \
                           - cos(omega*(time_1+delta)) \
                           - omega * delta * sin(omega*time_1)) \
                        / tanh(omega/(2*temperature))

def _2d_upper_triangle_integrand_imag(
        omega: ArrayLike,
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for imaginary part of upper triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (sin(omega*time_1) \
                            - sin(omega*(time_1+delta)) \
                            + omega * delta * cos(omega*time_1))

def _2d_lower_triangle_integrand_real(
        omega: ArrayLike,
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at zero
    temperature without J(omega). """
    return 1.0/omega**2 * (cos(omega*time_1) \
                           - cos(omega*(time_1-delta)) \
                           + omega * delta * sin(omega*time_1))

def _2d_lower_triangle_integrand_real_t(
        omega: ArrayLike,
        time_1: float,
        delta: float,
        temperature: float) -> ArrayLike:
    """Integrand for real part of lower triangle 2D time integral at finite
    temperature without J(omega). """
    return 1.0/omega**2 * (cos(omega*time_1) \
                           - cos(omega*(time_1-delta)) \
                           + omega * delta * sin(omega*time_1)) \
                        / tanh(omega/(2*temperature))

def _2d_lower_triangle_integrand_imag(
        omega: ArrayLike,
        time_1: float,
        delta: float) -> ArrayLike:
    """Integrand for imaginary part of lower triangle 2D time integral without
    J(omega). """
    return -1.0/omega**2 * (sin(omega*time_1)
                            - sin(omega*(time_1-delta))
                            - omega * delta * cos(omega*time_1))

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
    }

# --- spectral density classes ------------------------------------------------

class BaseSD(BaseAPIClass):
    """Base class for spectral densities."""

    def spectral_density(self, omega: ArrayLike) -> ArrayLike:
        r"""
        The resulting spectral density (including the cutoff).

        Parameters
        ----------
        omega : array_like
            The frequency :math:`\omega` for which we want to know the
            spectral density.

        Returns
        -------
        spectral_density : array_like
            The resulting spectral density :math:`J(\omega)` at the frequency
            :math:`\omega`.
        """
        raise NotImplementedError(
            "{} has no spectral_density implementation.".format(
                type(self).__name__))

    def correlation(
            self,
            tau: ArrayLike,
            temperature: Optional[float] = 0.0,
            epsrel: Optional[float] = 2**(-26)) -> ArrayLike:
        r"""
        Auto-correlation function associated to the spectral density at
        `temperature` :math:`T`

        .. math::

            C(\tau) = \int_0^\infty J(\omega)
            \left( \coth(\omega / 2T) \cos(\omega \tau)
            - i \sin(\omega \tau) \right) d\omega ,

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : array_like
            Time difference :math:`\tau`
        temperature : float
            The temperature :math:`T`.
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.

        Returns
        -------
        correlation : array_like
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        raise NotImplementedError(
            "{} has no correlation implementation.".format(type(self).__name__))

    def correlation_2d_integral(
            self,
            time_1: float,
            delta: float,
            temperature: Optional[float] = 0.0,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = 2**(-26)) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta(t_1,\Delta)_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta(t_1,\Delta)_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta(t_1,\Delta)_\mathrm{lower-triangle} =
            \int_{0}^{\Delta} \int_{t_1}^{t_1+dt''} C(t'-t'') dt' dt''

        for `shape` either ``'square'``, ``'upper-triangle'`` or
        ``'lower-triangle'``.

        Parameters
        ----------
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        delta : float
            Length of integration intevals.
        temperature : float
            The temperature :math:`T`.
        shape : str{``'square'``, ``'upper-triangle'``, ``'lower-triangle'``}
            The shape of the 2D integral.
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta(t_1,\Delta)_\mathrm{shape}`.
        """
        raise NotImplementedError(
            "{} has no correlation_2d_integral implementation.".format(
                type(self).__name__))


class CustomFunctionSD(BaseSD):
    r"""
    Spectral density with a custom function and a cutoff. The resulting
    spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c)

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
    cutoff_type : str{``'hard'``, ``'exponential'``, ``'gaussian'``}
        The cutoff type.
    """
    # Attributes
    # ----------
    # j_function : vectorized callable
    #     The spectral density without the cutoff.
    # cutoff : float
    #     The cutoff frequency.
    # cutoff_type : str{'hard','exponential','gaussian'}
    #     The cutoff type.

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            cutoff_type: Text = 'exponential',
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create spectral density with a custom function and a cutoff. """

        # check input: j_function
        try:
            __j_function = vectorize(j_function)
            float(__j_function(1.0))
        except:
            raise AssertionError(
                "Spectral density must be vectorizable and must return float.")
        self.j_function = __j_function

        # check input: cutoff
        try:
            __cutoff = float(cutoff)
        except:
            raise AssertionError("Cutoff must be a float.")
        self.cutoff = __cutoff

        # check input: cutoff_type
        assert cutoff_type in CUTOFF_DICT, \
            "Cutoff type must be one of: {}".format(CUTOFF_DICT.keys())
        self.cutoff_type = cutoff_type

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
        return "".join(ret)

    def spectral_density(self, omega: ArrayLike) -> ArrayLike:
        r"""
        The resulting spectral density (including the cutoff).

        Parameters
        ----------
        omega : array_like
            The frequency :math:`\omega` for which we want to know the
            spectral density.

        Returns
        -------
        spectral_density : array_like
            The resulting spectral density :math:`J(\omega)` at the frequency
            :math:`\omega`.
        """
        return self._spectral_density(omega)


    def correlation(
            self,
            tau: ArrayLike,
            temperature: Optional[float] = 0.0,
            epsrel: Optional[float] = 2**(-26)) -> ArrayLike:
        r"""
        Auto-correlation function associated to the spectral density at
        `temperature` :math:`T`

        .. math::

            C(\tau) = \int_0^\infty J(\omega)
            \left( \coth(\omega / 2T) \cos(\omega \tau)
            - i \sin(\omega \tau) \right) d\omega ,

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : array_like
            Time difference :math:`\tau`
        temperature : float
            The temperature :math:`T`.
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.

        Returns
        -------
        correlation : array_like
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if temperature == 0.0:
            re_integrand = lambda w: self._spectral_density(w) * cos(w*tau)
        else:
            re_integrand = lambda w: self._spectral_density(w) * cos(w*tau) \
                           / tanh(w/(2.0*temperature))
        im_integrand = lambda w: -1.0 * self._spectral_density(w) * sin(w*tau)
        # real and imaginary part of the integral
        re_int = integrate_semi_infinite(re_integrand,
                                         epsrel=epsrel)
        im_int = integrate_semi_infinite(im_integrand,
                                         epsrel=epsrel)
        return re_int+1j*im_int

    def correlation_2d_integral(
            self,
            time_1: float,
            delta: float,
            temperature: Optional[float] = 0.0,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = 2**(-26)) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta(t_1,\Delta)_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta(t_1,\Delta)_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta(t_1,\Delta)_\mathrm{lower-triangle} =
            \int_{0}^{\Delta} \int_{t_1}^{t_1+dt''} C(t'-t'') dt' dt''

        for `shape` either ``'square'``, ``'upper-triangle'`` or
        ``'lower-triangle'``.

        Parameters
        ----------
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        delta : float
            Length of integration intevals.
        temperature : float
            The temperature :math:`T`.
        shape : str{``'square'``, ``'upper-triangle'``, ``'lower-triangle'``}
            The shape of the 2D integral.
        epsrel : float (default = 1.49e-08)
            Relative error tollerance.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta(t_1,\Delta)_\mathrm{shape}`.
        """
        # real and imaginary part of the integrand
        if temperature == 0.0:
            re_integrand = lambda w: \
                    self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][0](w, time_1, delta)
        else:
            re_integrand = lambda w: \
                    self._spectral_density(w) \
                    * INTEGRAND_DICT[shape][1](w, time_1, delta, temperature)
        im_integrand = lambda w: self._spectral_density(w) \
                       * INTEGRAND_DICT[shape][2](w, time_1, delta)
        # real and imaginary part of the integral
        re_int = integrate_semi_infinite(re_integrand,
                                         epsrel=epsrel)
        im_int = integrate_semi_infinite(im_integrand,
                                         epsrel=epsrel)
        return re_int+1j*im_int


class StandardSD(CustomFunctionSD):
    r"""
    Spectral density of the standard form

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
        environment). The bath is called *ohmic* if :math:`\zeta=1`, *superohmic*
        if :math:`\zeta>1` and *subohmic* if :math:`\zeta<1`
    cutoff : float
        The cutoff frequency :math:`\omega_c`.
    cutoff_type : str{``'hard'``, ``'exponential'``, ``'gaussian'``}
        The cutoff type.
    """

    def __init__(
            self,
            alpha: float,
            zeta: float,
            cutoff: float,
            cutoff_type: Text = 'exponential',
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a standard spectral density."""

        # check input: alpha
        try:
            __alpha = float(alpha)
        except:
            raise AssertionError("Alpha must be a float.")
        self.alpha = __alpha

        # check input: zeta
        try:
            __zeta = float(zeta)
        except:
            raise AssertionError("Nu must be a float.")
        self.zeta = __zeta

        # check input: cutoff
        try:
            __cutoff = float(cutoff)
        except:
            raise AssertionError("Cutoff must be a float.")
        self.cutoff = __cutoff

        # use parent class for all the rest.
        j_function = lambda w: 2.0 * alpha * w**zeta * self.cutoff**(1-zeta)
        super().__init__(j_function,
                         cutoff=cutoff,
                         cutoff_type=cutoff_type,
                         name=name,
                         description=description,
                         description_dict=description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  alpha         = {} \n".format(self.alpha))
        ret.append("  zeta          = {} \n".format(self.zeta))

        return "".join(ret)

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

from typing import Callable, Optional, Union, Text
from typing import Any as ArrayLike
from functools import lru_cache

import numpy as np
from scipy import integrate

from oqupy.base_api import BaseAPIClass
from oqupy.config import INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.util import check_true

#np.seterr(all='warn')
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
            Relative error tolerance.
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

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
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
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            correlation_function: Callable[[float], float],
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Creates a CustomCorrelations object. """

        # check input: j_function
        try:
            tmp_correlation_function = np.vectorize(correlation_function)
            complex(tmp_correlation_function(1.0))
        except Exception as e:
            raise AssertionError("Correlation function must be vectorizable " \
                                 + "and must return float.") from e
        self.correlation_function = tmp_correlation_function

        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
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
            Relative error tolerance (has no effect here).
        subdiv_limit : int
            Maximal number of interval subdivisions for numerical integration
            (has no effect here).

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        return self.correlation_function(tau)

    @lru_cache(maxsize=2 ** 10, typed=False)
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

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        c_real = lambda y, x: np.real(self.correlation(x - y))
        c_imag = lambda y, x: np.imag(self.correlation(x - y))

        if time_2 is None:
            time_2 = time_1 + delta
        else:
            assert shape == 'rectangle', \
                "parameter 'time_2' can only be used in conjunction with " \
                "'shape' = ``rectangle`` !"

        lower_boundary = {'square': lambda x: 0.0,
                          'upper-triangle': lambda x: 0.0,
                          'rectangle': lambda x: 0.0}
        upper_boundary = {'square': lambda x: delta,
                          'upper-triangle': lambda x: x - time_1,
                          'rectangle': lambda x: delta, }

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
        return int_real + 1.0j * int_imag


# === CORRELATIONS FROM SPECTRAL DENSITIES ====================================


# --- the cutoffs -------------------------------------------------------------

def _hard_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)


def _exponential_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Exponential cutoff function."""
    return np.exp(-omega / omega_c)


def _gaussian_cutoff(omega: ArrayLike, omega_c: float) -> ArrayLike:
    """Gaussian cutoff function."""
    return np.exp(-(omega / omega_c) ** 2)


# dictionary for the various cutoffs in the form:
#   'cutoff_name': cutoff_function
CUTOFF_DICT = {
    'hard': _hard_cutoff,
    'exponential': _exponential_cutoff,
    'gaussian': _gaussian_cutoff,
}


# --- the spectral density classes --------------------------------------------

def _weighted_integral(
        re_integrand: Callable[[float], float],
        im_integrand: Callable[[float], float],
        w: float,
        a: Optional[float] = 0.0,
        b: Optional[float] = 1.0,
        epsrel: Optional[float] = INTEGRATE_EPSREL,
        limit: Optional[int] = SUBDIV_LIMIT) -> complex:
    re_int = integrate.quad(re_integrand,
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit,
                            weight='cos',
                            wvar=w)[0]
    im_int = integrate.quad(im_integrand,
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit,
                            weight='sin',
                            wvar=w)[0]
    return re_int + 1j * im_int

def _complex_integral(
        integrand: Callable[[float], complex],
        a: Optional[float] = 0.0,
        b: Optional[float] = 1.0,
        epsrel: Optional[float] = INTEGRATE_EPSREL,
        limit: Optional[int] = SUBDIV_LIMIT) -> complex:
    re_int = integrate.quad(lambda x: np.real(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]
    im_int = integrate.quad(lambda x: np.imag(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]

    return re_int + 1j * im_int

class CustomSD(BaseCorrelations):
    r"""
    Correlations corresponding to a custom spectral density for a thermal
    system with known temperature. The resulting spectral density is

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
    alt_integrator: bool
        Whether or not to use a sine/cosine weighted version of the
        integrals that could be better suited if you're encountering
        numerical difficulties, especially for long memory times.
        The integral computed in `eta_function` is cut in two: one
        handling the divergent part near 0 using the default integrator
        and the other taking care of the fast oscillations at angular
        frequency :math:`\tau` using an appropriate integration scheme.
        The cutoff point can be modified with `num_oscillations`.
    num_oscillations: int, float, callable
        When using `alt_integrator`, specifies how many oscillations to keep
        for the non-weighted part of the integral of `eta_function`. This should
        be either an integer or specified as a function of :math:`\tau`. To help
        choose: the higher this number is, the more the default integrator will
        struggle to integrate the non-weighted part. The lower it is, the more
        the weighted integrators will struggle to integrate the near divergent
        part close to 0. A constant value is a good choice for most use cases.
        Otherwise, it is possible to input a function to target possible issues
        at specific memory times :math:`\tau`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            cutoff_type: Optional[Text] = 'exponential',
            temperature: Optional[float] = 0.0,
            alt_integrator: Optional[bool] = False,
            num_oscillations: Optional[
                Union[int, float, Callable[[float], float]]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a CustomFunctionSD (spectral density) object. """

        # check input: j_function
        try:
            tmp_j_function = np.vectorize(j_function)
            float(tmp_j_function(1.0))
        except Exception as e:
            raise AssertionError("Spectral density must be vectorizable " \
                                 + "and must return float.") from e
        self.j_function = tmp_j_function

        # check input: cutoff
        try:
            tmp_cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = tmp_cutoff

        # check input: cutoff_type
        assert cutoff_type in CUTOFF_DICT, \
            "Cutoff type must be one of: {}".format(CUTOFF_DICT.keys())
        self.cutoff_type = cutoff_type

        # input check for temperature.
        try:
            tmp_temperature = float(temperature)
        except Exception as e:
            raise AssertionError("Temperature must be a float.") from e
        if tmp_temperature < 0.0:
            raise ValueError("Temperature must be >= 0.0 (but is {})".format(
                tmp_temperature))
        self.temperature = tmp_temperature

        # input check for alt_integrator
        assert isinstance(alt_integrator, bool)
        self._alt_integrator = alt_integrator

        # create the first_cutoff function and input check for num_oscillations
        if num_oscillations is None:
            num_oscillations = 3
        if isinstance(num_oscillations, (int, float)) and num_oscillations > 0:
            tmp_n = num_oscillations
            # This form is a step version of the first cutoff which prevents
            # useless computations of _truncated_eta
            self.first_cutoff = lambda t: 2*np.pi*tmp_n/(1+1/tmp_n)**np.floor(
                    np.log(t)/np.log(1+1/tmp_n))
        else:
            try:
                float(num_oscillations(1.0))
            except Exception as e:
                raise AssertionError("num_oscillations must be a positive" \
                    "float, int or a function returning floats or ints.") from e
            self.first_cutoff = lambda t: max(min(2*np.pi*num_oscillations(t)/t,
                                                  self.cutoff), 0.0)

        self._cutoff_function = \
            lambda omega: CUTOFF_DICT[self.cutoff_type](omega, self.cutoff)
        self._spectral_density = \
            lambda omega: self.j_function(omega) * self._cutoff_function(omega)

        super().__init__(name, description)

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
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            matsubara: Optional[bool] = False) -> ArrayLike:
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
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.

        Returns
        -------
        correlation : ndarray
            The auto-correlation function :math:`C(\tau)` at time :math:`\tau`.
        """
        # real and imaginary part of the integrand
        if matsubara:
            tau = -1j * tau
        # convention is tau.imag < 0
        if self.temperature == 0.0:
            check_true(
                matsubara is False,
                'Matsubara correlations only defined for temperature > 0')
            def integrand(w):
                return self._spectral_density(w) * np.exp(-1j * w * tau)
        else:
            def integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) \
                        * (np.exp(-1j * tau * w)
                           + np.exp(-(1 / self.temperature * w \
                                      - 1j * tau * w))) \
                        / (1 - np.exp(-w / self.temperature))
                else:
                    inte = self._spectral_density(w) * np.exp(-1j * w * tau)
                return inte

        integral = _complex_integral(integrand,
                                     a=0.0,
                                     b=self.cutoff,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)

        if self.cutoff_type != "hard":
            integral += _complex_integral(integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=subdiv_limit)
        if matsubara:
            integral = integral.real
        return integral

    @lru_cache(maxsize=2 ** 10, typed=False)
    def _truncated_eta(self,
                a: float,
                epsrel: float,
                limit: int) ->  complex:
        """Computes the parts of `eta_function` that are tau independant when
        `alt_integrator` is set to True.
        """
        if self.temperature == 0.0:
            def re_integrand(w):
                return self._spectral_density(w) / w ** 2
        else:
            def re_integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) / w ** 2 \
                        * 1/np.tanh(w / (2*self.temperature))
                else:
                    inte = self._spectral_density(w) / w ** 2
                return inte

        def im_integrand(w):
            return self._spectral_density(w)/w

        im_constant = integrate.quad(im_integrand,
                                     a=a,
                                     b=self.cutoff,
                                     epsrel=epsrel,
                                     limit=limit)[0]

        re_constant = integrate.quad(re_integrand,
                                         a=a,
                                         b=self.cutoff,
                                         epsrel=epsrel,
                                         limit=limit)[0]

        if self.cutoff_type != "hard":
            im_constant += integrate.quad(im_integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=limit)[0]

            re_constant += integrate.quad(re_integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=limit)[0]
        return re_constant, im_constant

    @lru_cache(maxsize=2 ** 10, typed=False)
    def eta_function(
            self,
            tau: ArrayLike,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            alt_integrator: Optional[Union[bool, None]] = None,
            matsubara: Optional[bool] = False) -> ArrayLike:
        r"""
        :math:`\eta` function associated to the spectral density at the
        given temperature :math:`T` as given in [Strathearn2017]

        .. math::

            \eta(\tau) = \int_0^{\infty} \frac{J(\omega)}{\omega^2} \
                       \left[ ( \cos(\omega \tau) - 1 ) \
                              \coth\left( \frac{\omega}{2 T}\right) \
                              - i ( \sin(\omega \tau) - \omega \tau ) \right] \
                              \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.

        Parameters
        ----------
        tau : ndarray
            Time difference :math:`\tau`
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.
        alt_integrator: Union[bool, None]
            Whether or not to use a sine/cosine weighted version of the
            integrals that could be better suited if you're encountering
            numerical difficulties, especially for long memory times. If the
            value is `None`, the value of `alt_integrator` set during
            initialization of the `CustomSD` object is used instead.
        Returns
        -------
        correlation : ndarray
            The function :math:`\eta(\tau)` at time :math:`\tau`.
        """
        if alt_integrator is None:
            alt_integrator = self._alt_integrator
        check_true(not (matsubara is True and alt_integrator is True),
                   "Can't use weighted integrator with matsubara")
        check_true(isinstance(alt_integrator, bool),
                   "alt_integrator has to be either True, False or None")
        # real and imaginary part of the integrand
        if matsubara:
            tau = -1j * tau
        # convention is tau.imag < 0
        if self.temperature == 0.0:
            check_true(
                matsubara is False,
                'Matsubara correlations only defined for temperature > 0')
            def integrand(w):
                return self._spectral_density(w) / w ** 2 * (
                    (np.exp(-1j * w * tau) - 1) + 1j * w * tau)
        else:
            def integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (((np.exp(-1j*tau * w) \
                             + np.exp(-(w / self.temperature - 1j*tau * w))) \
                            - np.exp(- w / self.temperature) - 1) \
                        / (1 - np.exp(-w / self.temperature)) + 1j*tau * w)
                else:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (np.exp(-1j * w * tau) - 1 + 1j * w * tau)
                return inte

        # Alternative computation with weighted integrators
        if alt_integrator and tau*self.cutoff > 1.:
            im_integrand = lambda w: - self._spectral_density(w) / w ** 2
            if self.temperature == 0.0:
                re_integrand = lambda w: self._spectral_density(w) / w ** 2
            else:
                def re_integrand(w):
                    # this is to stop overflow
                    if np.exp(-w / self.temperature) > np.finfo(float).eps:
                        inte = self._spectral_density(w) / w ** 2 \
                            * 1/np.tanh(w / (2*self.temperature))
                    else:
                        inte = self._spectral_density(w) / w ** 2
                    return inte

            omega_tilde = self.first_cutoff(tau)

            integral = _complex_integral(integrand,
                                         a=0.0,
                                         b=omega_tilde,
                                         epsrel=epsrel,
                                         limit=subdiv_limit)

            integral += _weighted_integral(re_integrand, im_integrand,
                                         w=tau,
                                         a=omega_tilde,
                                         b=self.cutoff,
                                         epsrel=epsrel,
                                         limit=subdiv_limit)

            if self.cutoff_type != "hard":
                integral += _weighted_integral(re_integrand, im_integrand,
                                              w=tau,
                                              a=self.cutoff,
                                              b=np.inf,
                                              epsrel=epsrel,
                                              limit=subdiv_limit)

            re_constant, im_constant = self._truncated_eta(a=omega_tilde,
                                                           epsrel=epsrel,
                                                           limit=subdiv_limit)
            return - (integral + 1.j*tau*im_constant - re_constant)

        # Default computation of eta_function when alt_integrator is False
        integral = _complex_integral(integrand,
                                     a=0.0,
                                     b=self.cutoff,
                                     epsrel=epsrel,
                                     limit=subdiv_limit)

        if self.cutoff_type != "hard":
            integral += _complex_integral(integrand,
                                          a=self.cutoff,
                                          b=np.inf,
                                          epsrel=epsrel,
                                          limit=subdiv_limit)
        if matsubara:
            integral = integral.real
        return -integral

    def correlation_2d_integral(
            self,
            delta: float,
            time_1: float,
            time_2: Optional[float] = None,
            shape: Optional[Text] = 'square',
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            alt_integrator: Optional[Union[bool, None]] = None,
            matsubara: Optional[bool] = False) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        Parameters
        ----------
        delta : float
            Length of integration intervals.
        time_1 : float
            Lower bound of integration interval of :math:`dt'`.
        time_2 : float
            Upper bound of integration interval of :math:`dt'` for `shape` =
            ``'rectangle'``.
        shape : str (default = ``'square'``)
            The shape of the 2D integral. Shapes are: {``'square'``,
            ``'upper-triangle'``, ``'rectangle'``}
        epsrel : float
            Relative error tolerance.
        subdiv_limit: int
            Maximal number of interval subdivisions for numerical integration.
        alt_integrator: Union[bool, None]
            Whether or not to use a sine/cosine weighted version of the
            integrals that could be better suited if you're encountering
            numerical difficulties, especially for long times.

        Returns
        -------
        integral : float
            The numerical value for the two dimensional integral
            :math:`\eta_\mathrm{shape}`.
        """
        kwargs = {
            'epsrel': epsrel,
            'subdiv_limit': subdiv_limit,
            'alt_integrator': alt_integrator,
            'matsubara': matsubara}

        if shape == 'upper-triangle':
            integral = self.eta_function(time_1 + delta, **kwargs) \
                       - self.eta_function(time_1, **kwargs)
        elif shape == 'square':
            integral = self.eta_function(time_1 + delta, **kwargs) \
                       - 2.0 * self.eta_function(time_1, **kwargs) \
                       + self.eta_function(time_1 - delta, **kwargs)
        elif shape == 'rectangle':
            integral = self.eta_function(time_2, **kwargs) \
                       - self.eta_function(time_1, **kwargs) \
                       - self.eta_function(time_2 - delta, **kwargs) \
                       + self.eta_function(time_1 - delta, **kwargs)
        else:
            raise NotImplementedError("Shape '{shape}' not implemented.")

        if matsubara:
            integral = integral.real
        return integral


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
        The coupling strength :math:`\alpha`.
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
    alt_integrator: bool
        Whether or not to use a sine/cosine weighted version of the
        integrals that could be better suited if you're encountering
        numerical difficulties, especially for long memory times.
        The integral computed in `eta_function` is cut in two: one
        handling the divergent part near 0 using the default integrator
        and the other taking care of the fast oscillations at angular
        frequency :math:`\tau` using an appropriate integration scheme.
        The cutoff point can be modified with `num_oscillations`.
    num_oscillations: int, float, callable
        When using `alt_integrator`, specifies how many oscillations to keep
        for the non-weighted part of the integral of `eta_function`. This should
        be either an integer or specified as a function of :math:`\tau`. To help
        choose: the higher this number is, the more the default integrator will
        struggle to integrate the non-weighted part. The lower it is, the more
        the weighted integrators will struggle to integrate the near divergent
        part close to 0. A constant value is a good choice for most use cases.
        Otherwise, it is possible to input a function to target possible issues
        at specific memory times :math:`\tau`.
    name: str
        An optional name for the correlations.
    description: str
        An optional description of the correlations.
    """

    def __init__(
            self,
            alpha: float,
            zeta: float,
            cutoff: float,
            cutoff_type: Text = 'exponential',
            temperature: Optional[float] = 0.0,
            alt_integrator: Optional[bool] = False,
            num_oscillations: Optional[
                Union[int, float, Callable[[float], float]]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a StandardSD (spectral density) object. """

        # check input: alpha
        try:
            tmp_alpha = float(alpha)
        except Exception as e:
            raise AssertionError("Alpha must be a float.") from e
        self.alpha = tmp_alpha

        # check input: zeta
        try:
            tmp_zeta = float(zeta)
        except Exception as e:
            raise AssertionError("Nu must be a float.") from e
        self.zeta = tmp_zeta

        # check input: cutoff
        try:
            tmp_cutoff = float(cutoff)
        except Exception as e:
            raise AssertionError("Cutoff must be a float.") from e
        self.cutoff = tmp_cutoff

        # use parent class for all the rest.
        j_function = lambda w: 2.0 * self.alpha * w ** self.zeta \
                               * self.cutoff ** (1 - zeta)

        super().__init__(j_function,
                         cutoff=cutoff,
                         cutoff_type=cutoff_type,
                         temperature=temperature,
                         alt_integrator=alt_integrator,
                         name=name,
                         description=description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  alpha         = {} \n".format(self.alpha))
        ret.append("  zeta          = {} \n".format(self.zeta))

        return "".join(ret)

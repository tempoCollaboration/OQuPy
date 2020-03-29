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
Numerical integration functions for TEMPO.
"""

import warnings
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import quad

from time_evolving_mpo.exceptions import NumericsWarning
from time_evolving_mpo.exceptions import NumericsError


# --- integrate function for interval 0 to infty ------------------------------

def integrate_semi_infinite(
        integrand: Callable,
        epsrel: Optional[float] = 2**-26) -> Tuple[float, float]:
    r"""
    Numerical integration from 0 to infinity. This function performs the
    numerical integration of

    .. math::

        \int_0^\infty f(x) dx ,

    with `integrand` :math:`f(x)` and  `cutoff` :math:`x_c`.

    Parameters
    ----------
    integrand : vectorized callable
        A numpy-vectorized callable function.
    epsrel : float (default = 1.49e-08)
        Relative error tollerance.

    Returns
    -------
    integral : float
        The integral.

    Raises
    ------
    `NumericsError` :
        If anything goes wrong during the integration.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = quad(integrand,
                          0.0,
                          np.inf,
                          epsabs=0.0,
                          epsrel=epsrel,
                          limit=2**10)
        except Exception as error:
            raise NumericsError("Integration error: \n{}".format(error))
    return result[0]


# --- integrate gauss laguerre ------------------------------------------------


@lru_cache(2**8)
def _gauss_laguerre_quad(deg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the positions and weights of the Gauss-Laguerre qudrature. """
    return np.polynomial.laguerre.laggauss(deg)


def gauss_laguerre(
        integrand: Callable,
        deg: int,
        rescale: Optional[float] = 1.0) -> float:
    r"""
    Integration with Gauss-Laguerre quadrature of fixed degree. This function
    performs the numerical integration of

    .. math::

        \int_0^\infty f(x) \exp\left[ -x/x_c \right] dx ,

    with `integrand` :math:`f(x)` and  `rescale` :math:`x_c` employing the
    Gauss-Laguerre quadrature of degree `deg`.

    Parameters
    ----------
    integrand : vectorized callable
        A numpy-vectorized callable function.
    deg : int
        The degree of Laguerre polynominals.
    rescale : float
        The rescaling of the exponential :math:`x_c`.

    Returns
    -------
    interal : float
        The integral.
    """
    positions, weights = _gauss_laguerre_quad(deg)
    return rescale*np.sum(integrand(positions*rescale)*weights)


def gauss_laguerre_adaptive(
        integrand: Callable,
        rescale: Optional[float] = 1.0,
        epsrel: Optional[float] = 2**-26,
        min_deg: Optional[int] = 10,
        warn_deg: Optional[int] = 100,
        max_deg: Optional[int] = 150) -> Tuple[float, float]:
    r"""
    Adaptive integration with Gauss-Laguerre quadrature. This function performs
    the numerical integration of

    .. math::

        \int_0^\infty f(x) \exp\left[ -x/x_c \right] dx ,

    with `integrand` :math:`f(x)` and  `rescale` :math:`x_c` employing the
    Gauss-Laguerre quadrature.

    Parameters
    ----------
    integrand : vectorized callable
        A numpy-vectorized callable function.
    rescale : float
        The rescaling of the exponential :math:`x_c`.
    epsrel : float (default = 1.49e-08)
        Relative error tollerance.
    min_deg : int (default = 10)
        Minimal degree of Laguerre polynominals.
    warn_deg : int (default = 100)
        Warn with NumericsWarning when degree of Laguerre polynominals exceeds
        `warn_deg`
    max_deg : int (default = 150)
        Maximal degree of Laguerre polynominals.

    Returns
    -------
    integral : float
        The integral.

    Raises
    ------
    `NumericsError`
        If anything goes wrong during the integration.
    """
    deg = min_deg
    current_result = gauss_laguerre(integrand,
                                    deg,
                                    rescale=rescale)

    do_while_condition = True
    while do_while_condition:
        deg += 1
        if deg == warn_deg:
            warnings.warn("Gauss Laguerre integration is on it's edge!",
                          NumericsWarning)
        if deg == max_deg + 1:
            raise NumericsError(
                "Gauss Laguerre integration can't reach "
                + "relative precission of {}.".format(epsrel))
        previous_result = current_result
        current_result = gauss_laguerre(integrand,
                                        deg,
                                        rescale=rescale)
        abs_error = abs(current_result - previous_result)
        rel_error = abs(abs_error / current_result)
        if rel_error <= epsrel:
            do_while_condition = False

    return current_result

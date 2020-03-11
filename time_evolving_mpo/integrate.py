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

from time_evolving_mpo import NumericsWarning
from time_evolving_mpo import NumericsError


# ------- integrate function with hard cutoff -------------------------------

def semi_infinite_hard_cutoff(
        integrand: Callable,
        cutoff: float,
        epsrel: Optional[float] = 2**-26) -> Tuple[float, float]:
    r""" Numerical integration from 0 to a hard cutoff.

    This function performs the numerical integration of:
    :math:`\int_0^{x_c} f(x) dx`, with
    *integrand* :math:`f(x)` and  *cutoff* :math:`x_c`.

    Args:
        integrand: A numpy-vectorizable callable function.
        cutoff: The upper limit of the integration interval.
        epsrel: Relative error tollerance.

    Returns:
        The integral and an estimate of the absolute numerical error.

    Raises:
        NumericsError: If anything goes wrong during the integration.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = quad(integrand,
                          0.0,
                          cutoff,
                          epsabs=0.0,
                          epsrel=epsrel,
                          limit=2**10)
        except Exception as error:
            raise NumericsError("Integration error: \n{}".format(error))
    return result


# ------- integrate function with gaussian cutoff ----------------------------

def semi_infinite_gaussian_cutoff(
        integrand: Callable,
        cutoff: float,
        epsrel: Optional[float] = 2**-26) -> Tuple[float, float]:
    r""" Numerical integration from 0 to infinity with a exponential cutoff.

    This function performs the numerical integration of
    :math:`\int_0^\inf f(x)\exp\left[ -\left( x/x_c \right)^2 \right] dx`, with
    *integrand* :math:`f(x)` and  *cutoff* :math:`x_c`.

    Args:
        integrand: A numpy-vectorizable callable function.
        cutoff: The gaussian cutoff :math:`x_c`.
        epsrel: Relative error tollerance.

    Returns:
        The integral and an estimate of the absolute numerical error.

    Raises:
        NumericsError: If anything goes wrong during the integration.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            full_integrand = lambda w: integrand(w) * np.exp(-(w/cutoff)**2)
            result = quad(full_integrand,
                          0.0,
                          np.inf,
                          epsabs=0.0,
                          epsrel=epsrel,
                          limit=2**10)
        except Exception as error:
            raise NumericsError("Integration error: \n{}".format(error))
    return result


# ------- integrate function with exponential cutoff --------------------------

def semi_infinite_exponential_cutoff(
        integrand: Callable,
        cutoff: float,
        epsrel: Optional[float] = 2**-26) -> Tuple[float, float]:
    r""" Numerical integration from 0 to infinity with a gaussian cutoff.

    This function performs the numerical integration of
    :math:`\int_0^\inf f(x) \exp\left[ -x/x_c \right] dx`, with
    *integrand* :math:`f(x)` and  *cutoff* :math:`x_c`.

    Args:
        integrand: A numpy-vectorizable callable function.
        cutoff: The exponential cutoff :math:`x_c`.
        epsrel: Relative error tollerance.

    Returns:
        The integral and an estimate of the absolute numerical error.

    Raises:
        NumericsError: If anything goes wrong during the integration.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            full_integrand = lambda w: integrand(w) * np.exp(-w/cutoff)
            result = quad(full_integrand,
                          0.0,
                          np.inf,
                          epsabs=0.0,
                          epsrel=epsrel,
                          limit=2**10)
        except Exception as error:
            raise NumericsError("Integration error: \n{}".format(error))
    return result


@lru_cache(2**8)
def _gauss_laguerre_quad(deg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the positions and weights of the Gauss-Laguerre qudrature."""
    return np.polynomial.laguerre.laggauss(deg)


def gauss_laguerre(
        integrand: Callable,
        deg: int,
        rescale: Optional[float] = 1.0) -> float:
    r"""Integration with Gauss-Laguerre quadrature of fixed degree.

    This function performs the numerical integration of
    :math:`\int_0^\inf f(x) \exp\left[ -x/x_c \right] dx`, with
    *integrand* :math:`f(x)` and  *rescale* :math:`x_c` employing the
    Gauss-Laguerre quadrature of degree *deg*.

    Args:
        integrand: A numpy-vectorizable callable function.
        deg: The degree of Laguerre polynominals.
        rescale: The rescaling of the exponential :math:`x_c`.

    Returns:
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
    r"""Adaptive integration with Gauss-Laguerre quadrature.

    This function performs the numerical integration of
    :math:`\int_0^\inf f(x) \exp\left[ -x/x_c \right] dx`, with
    *integrand* :math:`f(x)` and  *rescale* :math:`x_c` employing the
    Gauss-Laguerre quadrature.

    Args:
        integrand: A numpy-vectorizable callable function.
        rescale: The rescaling of the exponential :math:`x_c`.
        epsrel: Relative error tollerance.
        min_deg: Minimal degree of Laguerre polynominals.
        warn_deg: Warn with NumericsWarning when degree of Laguerre
            polynominals exceeds *warn_deg*
        max_deg: Maximal degree of Laguerre polynominals.

    Returns:
        The integral and an estimate of the absolute numerical error.

    Raises:
        NumericsError: If anything goes wrong during the integration.
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

    return current_result, abs_error

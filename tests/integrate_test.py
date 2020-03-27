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
Tests for the time_evovling_mpo.integrate module.
"""

import pytest
import numpy as np

from time_evolving_mpo.exceptions import NumericsError
from time_evolving_mpo.exceptions import NumericsWarning
from time_evolving_mpo.integrate import integrate_semi_infinite
from time_evolving_mpo.integrate import gauss_laguerre
from time_evolving_mpo.integrate import gauss_laguerre_adaptive

def f_1(x):
    return x**3

def f_2(x):
    return x**0.5

def f_3(x):
    return np.exp(np.sqrt(x))

def f_4(x):
    return np.exp(np.sqrt(x)) * np.sin(10.0*x)

def f_5(x):
    return np.exp(np.sqrt(x)) * np.sin(10.0*x) * np.cos(97.0*x)

list_exponential = [(f_1, 96.0),
                    (f_2, 2.5066282746310002),
                    (f_3, 8.954103623407388),
                    (f_4, 0.12093517077496016),
                   ]

# -- test semi infinite integrate ---------------------------------------------

@pytest.mark.parametrize("a",list_exponential)
def test_integrate_semi_infinite(a):
    integrand = lambda x : a[0](x) * np.exp(-x/2.0)
    result = integrate_semi_infinite(integrand,epsrel=2**(-20))
    np.testing.assert_almost_equal(result,a[1])

def test_integrate_semi_infinite():
    integrand = lambda x : f_5(x) * np.exp(-x/2.0)
    with pytest.raises(NumericsError):
        with pytest.warns(NumericsWarning):
            integrate_semi_infinite(integrand)


# -- test gauss laguerre ------------------------------------------------------

@pytest.mark.parametrize("a",list_exponential[:1])
def test_gauss_laguerre(a):
    result = gauss_laguerre(a[0], deg=3, rescale=2.0)
    np.testing.assert_almost_equal(result,a[1])

@pytest.mark.parametrize("a",list_exponential[:1])
def test_gauss_laguerre_adaptive(a):
    result = gauss_laguerre_adaptive(a[0], rescale=2.0)
    np.testing.assert_almost_equal(result,a[1])

def test_gauss_laguerre_adaptive_error():
    with pytest.raises(NumericsError):
        with pytest.warns(NumericsWarning):
            gauss_laguerre_adaptive(f_5, rescale=100.0)

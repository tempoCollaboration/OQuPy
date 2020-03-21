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
Tests for the time_evovling_mpo.spectral_density module.
"""

import pytest
import numpy as np

from time_evolving_mpo.spectral_density import BaseSD
from time_evolving_mpo import CustomFunctionSD
from time_evolving_mpo import StandardSD


square_function = lambda w: 0.1 * w**2

def test_base_sd():
    sd = BaseSD()
    str(sd)
    with pytest.raises(NotImplementedError):
        sd.spectral_density(None)
    with pytest.raises(NotImplementedError):
        sd.correlation(None,temperature=0.0)
    with pytest.raises(NotImplementedError):
        sd.correlation(None,temperature=2.0)
    for shape in ["square", "upper-triangle", "lower-triangle"]:
        with pytest.raises(NotImplementedError):
            sd.correlation_2d_integral(time_1=None,
                                       delta=None,
                                       temperature=2.0,
                                       shape="square")

def test_custom_function_sd():
    for cutoff_type in ["hard", "exponential", "gaussian"]:
        sd = CustomFunctionSD(square_function,
                              cutoff=2.0,
                              cutoff_type="gaussian")
        str(sd)
        w = np.linspace(0, 8.0*sd.cutoff, 10)
        y = sd.spectral_density(w)
        t = np.linspace(0, 4.0/sd.cutoff, 10)
        [sd.correlation(tt,temperature=0.0) for tt in t]
        [sd.correlation(tt,temperature=2.0) for tt in t]
        for shape in ["square", "upper-triangle", "lower-triangle"]:
            sd.correlation_2d_integral(time_1=0.25,
                                         delta=0.05,
                                         temperature=0.0,
                                         shape=shape)
            sd.correlation_2d_integral(time_1=0.25,
                                         delta=0.05,
                                         temperature=2.0,
                                         shape=shape)
def test_custom_function_sd_bad_input():
    with pytest.raises(AssertionError):
        CustomFunctionSD("o-ohh!", cutoff=2.0, cutoff_type="gaussian")
    with pytest.raises(AssertionError):
        CustomFunctionSD(square_function, cutoff=None, cutoff_type="gaussian")
    with pytest.raises(AssertionError):
        CustomFunctionSD(square_function, cutoff=2.0, cutoff_type="bla")


def test_standard_sd():
    for cutoff_type in ["hard", "exponential", "gaussian"]:
        sd = StandardSD(alpha=0.25,
                        zeta=1.0,
                        cutoff=2.0,
                        cutoff_type=cutoff_type)
        str(sd)
        w = np.linspace(0, 8.0*sd.cutoff, 10)
        y = sd.spectral_density(w)
        t = np.linspace(0, 4.0/sd.cutoff, 10)
        [sd.correlation(tt,temperature=0.0) for tt in t]
        [sd.correlation(tt,temperature=2.0) for tt in t]
        for shape in ["square", "upper-triangle", "lower-triangle"]:
            sd.correlation_2d_integral(time_1=0.25,
                                         delta=0.05,
                                         temperature=2.0,
                                         shape=shape)
            sd.correlation_2d_integral(time_1=0.25,
                                         delta=0.05,
                                         temperature=0.0,
                                         shape=shape)

def test_standard_sd_bad_input():
    with pytest.raises(AssertionError):
        StandardSD(alpha="bla", zeta=1.0, cutoff=2.0, cutoff_type="hard")
    with pytest.raises(AssertionError):
        StandardSD(alpha=0.25, zeta="bla", cutoff=2.0, cutoff_type="hard")
    with pytest.raises(AssertionError):
        StandardSD(alpha=0.25, zeta=1.0, cutoff="bla", cutoff_type="hard")

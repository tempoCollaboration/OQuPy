# Copyright 2022 The TEMPO Collaboration
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
Tests for the time_evovling_mpo.correlations module.
"""

import pytest
import numpy as np

from oqupy.correlations import BaseCorrelations
from oqupy.correlations import CustomCorrelations
from oqupy.correlations import CustomSD
from oqupy.correlations import PowerLawSD

square_function = lambda w: 0.1 * w**2

def test_base_correlations():
    cor = BaseCorrelations()
    str(cor)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    for shape in ["square", "upper-triangle", "lower-triangle"]:
        with pytest.raises(NotImplementedError):
            cor.correlation_2d_integral(time_1=None,
                                        delta=None,
                                        shape=shape)

def test_custom_correlations():
    correlation_fun = lambda w: (np.cos(w)+1j*np.sin(w*0.7)) * np.exp(-w/2)
    cor = CustomCorrelations(correlation_fun)
    str(cor)
    t = np.linspace(0, 4.0/15.0, 10)
    [cor.correlation(tt) for tt in t]
    for shape in ["square", "upper-triangle", "lower-triangle"]:
        cor.correlation_2d_integral(time_1=0.25,
                                    delta=0.05,
                                    shape=shape)
        cor.correlation_2d_integral(time_1=0.25,
                                    delta=0.05,
                                    shape=shape)

def test_custom_correlations_bad_input():
    with pytest.raises(AssertionError):
        cor = CustomCorrelations("ohh-o")


def test_custom_s_d():
    for cutoff_type in ["hard", "exponential", "gaussian"]:
        for temperature in [0.0, 2.0]:
            sd = CustomSD(square_function,
                          cutoff=2.0,
                          temperature=temperature,
                          cutoff_type=cutoff_type)
            str(sd)
            w = np.linspace(0, 8.0*sd.cutoff, 10)
            y = sd.spectral_density(w)
            t = np.linspace(0, 4.0/sd.cutoff, 10)
            [sd.correlation(tt) for tt in t]
            for shape in ["square", "upper-triangle", "lower-triangle"]:
                sd.correlation_2d_integral(time_1=0.25,
                                           delta=0.05,
                                           shape=shape)
                sd.correlation_2d_integral(time_1=0.25,
                                           delta=0.05,
                                           shape=shape)

def test_custom_s_d_bad_input():
    with pytest.raises(AssertionError):
        CustomSD("o-ohh!", cutoff=2.0, cutoff_type="gaussian", temperature=0.0)
    with pytest.raises(AssertionError):
        CustomSD(square_function, cutoff=None, cutoff_type="gaussian", \
                 temperature=0.0)
    with pytest.raises(AssertionError):
        CustomSD(square_function, cutoff=2.0, cutoff_type="bla", \
                 temperature=0.0)
    with pytest.raises(AssertionError):
        CustomSD(square_function, cutoff=2.0, cutoff_type="gaussian", \
                 temperature="bla")
    with pytest.raises(ValueError):
        CustomSD(square_function, cutoff=2.0, cutoff_type="hard", \
                 temperature=-2.0)


def test_power_law_s_d():
    for cutoff_type in ["hard", "exponential", "gaussian"]:
        sd = PowerLawSD(alpha=0.25,
                        zeta=1.0,
                        cutoff=2.0,
                        cutoff_type=cutoff_type,
                        temperature=2.0)
        str(sd)
        w = np.linspace(0, 8.0*sd.cutoff, 10)
        y = sd.spectral_density(w)
        t = np.linspace(0, 4.0/sd.cutoff, 10)
        [sd.correlation(tt) for tt in t]
        [sd.correlation(tt) for tt in t]
        for shape in ["square", "upper-triangle", "lower-triangle"]:
            sd.correlation_2d_integral(time_1=0.25,
                                       delta=0.05,
                                       shape=shape)
            sd.correlation_2d_integral(time_1=0.25,
                                       delta=0.05,
                                       shape=shape)

def test_power_law_s_d_bad_input():
    with pytest.raises(AssertionError):
        PowerLawSD(alpha="bla", zeta=1.0, cutoff=2.0, cutoff_type="hard")
    with pytest.raises(AssertionError):
        PowerLawSD(alpha=0.25, zeta="bla", cutoff=2.0, cutoff_type="hard")
    with pytest.raises(AssertionError):
        PowerLawSD(alpha=0.25, zeta=1.0, cutoff="bla", cutoff_type="hard")
    with pytest.raises(ValueError):
        PowerLawSD(alpha=0.25, zeta=1.0, cutoff=2.0, cutoff_type="hard", \
                   temperature=-2.0)

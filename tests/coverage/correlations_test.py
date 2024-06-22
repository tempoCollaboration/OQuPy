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

from oqupy.bath_correlations import BaseCorrelations
from oqupy.bath_correlations import CustomCorrelations
from oqupy.bath_correlations import CustomSD
from oqupy.bath_correlations import PowerLawSD

square_function = lambda w: 0.1 * w ** 2


def test_base_correlations():
    cor = BaseCorrelations()
    str(cor)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    with pytest.raises(NotImplementedError):
        cor.correlation(None)
    for shape in ["square", "upper-triangle"]:
        with pytest.raises(NotImplementedError):
            cor.correlation_2d_integral(time_1=None,
                                        delta=None,
                                        shape=shape)


def test_custom_correlations():
    correlation_fun = lambda w: (np.cos(w) + 1j * np.sin(w * 0.7)) * np.exp(-w / 2)
    cor = CustomCorrelations(correlation_fun)
    str(cor)
    t = np.linspace(0, 4.0 / 15.0, 10)
    [cor.correlation(tt) for tt in t]
    for shape in ["square", "upper-triangle"]:
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
            w = np.linspace(0, 8.0 * sd.cutoff, 10)
            y = sd.spectral_density(w)
            t = np.linspace(0, 4.0 / sd.cutoff, 10)
            [sd.correlation(tt) for tt in t]
            for shape in ["square", "upper-triangle"]:
                sd.correlation_2d_integral(time_1=0.25,
                                           delta=0.05,
                                           shape=shape)
                sd.correlation_2d_integral(time_1=0.25,
                                           delta=0.05,
                                           shape=shape)


def test_matsubara_custom_s_d():
    for cutoff_type in ["hard", "exponential", "gaussian"]:
        temperature = 2.0
        cutoff = 2.0
        j_function = square_function
        sd = CustomSD(j_function,
                      cutoff=cutoff,
                      temperature=temperature,
                      cutoff_type=cutoff_type)


        str(sd)
        w = np.linspace(0, 8.0 * sd.cutoff, 10)
        y = sd.spectral_density(w)
        t = np.linspace(0, 4.0 / sd.cutoff, 10)
        [sd.correlation(tt, matsubara=True) for tt in t]
        [sd.eta_function(tt, matsubara=True) for tt in t]

        assert type(sd.correlation(1, matsubara=True)) == float
        np.testing.assert_almost_equal(
            sd.correlation(0, matsubara=True),
            sd.correlation(1 / temperature, matsubara=True))

        for shape in ["square", "upper-triangle"]:
            sd.correlation_2d_integral(time_1=0.25,
                                       delta=0.05,
                                       shape=shape,
                                       matsubara=True)
            sd.correlation_2d_integral(time_1=0.25,
                                       delta=0.05,
                                       shape=shape,
                                       matsubara=True)

        dt = 0.25
        for ma in [True, False]:
            big_tri = sd.correlation_2d_integral(time_1=0.0,
                                                 delta=4*dt,
                                                 shape='upper-triangle',
                                                 matsubara=ma)
            mid_tri = sd.correlation_2d_integral(time_1=0.0,
                                                 delta=2*dt,
                                                 shape='upper-triangle',
                                                 matsubara=ma)
            small_tri = sd.correlation_2d_integral(time_1=0.0,
                                                   delta=dt,
                                                   shape='upper-triangle',
                                                   matsubara=ma)
            square = sd.correlation_2d_integral(time_1=2*dt,
                                                delta=dt,
                                                shape='square',
                                                matsubara=ma)
            rect = sd.correlation_2d_integral(time_1=2 * dt,
                                              time_2=4 * dt,
                                              delta=dt,
                                              shape='rectangle',
                                              matsubara=ma)
            # assert np.round(big_tri - (3 * small_tri + square + rect), 14) == 0
            stiched_big_tri = 3 * mid_tri + square + rect - 2*small_tri
            np.testing.assert_almost_equal( stiched_big_tri, big_tri)

def test_matsubara_custom_s_d_bad_input():
    temperature = 0.0
    correlations = CustomSD(square_function, cutoff=2.0, cutoff_type="gaussian", temperature=temperature)
    with pytest.raises(ValueError):
        correlations.correlation(1, matsubara=True)
    with pytest.raises(ValueError):
        correlations.eta_function(1, matsubara=True)
    for shape in ['square', 'upper-triangle', 'rectangle']:
        with pytest.raises(ValueError):
            correlations.correlation_2d_integral(time_1=2,
                                                  time_2=1,
                                                  delta=0.1,
                                                  shape=shape,
                                                  matsubara=True)




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
        w = np.linspace(0, 8.0 * sd.cutoff, 10)
        y = sd.spectral_density(w)
        t = np.linspace(0, 4.0 / sd.cutoff, 10)
        [sd.correlation(tt) for tt in t]
        [sd.correlation(tt) for tt in t]
        for shape in ["square", "upper-triangle"]:
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

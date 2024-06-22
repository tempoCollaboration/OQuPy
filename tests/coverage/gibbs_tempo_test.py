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
Tests for the Tempo module.
"""
import pytest
import numpy as np

import oqupy

import oqupy.tempo as tempo



def test_gibbs_tempo():

    coupling = 0.2
    temperature = 0.23
    cutoff = 5
    ohmicity = 1
    steps = 11

    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))

    jw = lambda w: 2 * coupling * w ** ohmicity / (cutoff ** (ohmicity - 1))

    correlations = oqupy.CustomSD(jw, cutoff, cutoff_type='exponential', temperature=temperature)

    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)

    gibbs_param = tempo.GibbsParameters(steps, 1.0e-5, name="rough-A")

    gibbs_sys = tempo.GibbsTempo(system=system,
                              bath=bath,
                              parameters=gibbs_param)

    system = oqupy.System(0.5 * oqupy.operators.sigma("x"), gammas=[1], lindblad_operators=[0.5 * oqupy.operators.sigma("x")])

    with pytest.raises(Warning):
        gibbs_sys = tempo.GibbsTempo(system=system,
                                     bath=bath,
                                     parameters=gibbs_param)

    assert gibbs_sys.dimension == 2
    assert gibbs_sys.temperature == temperature

    dyn1 = gibbs_sys.compute(progress_type="bar")
    dyn2 = gibbs_sys.get_dynamics()
    assert dyn1 == dyn2
    assert len(dyn1.times) == steps + 1  # should be steps + 1
    assert dyn1.times[-1] == 1 / temperature

    state1 = gibbs_sys.get_state()
    state2 = dyn1.states[-1]
    state2 = state2 / state2.trace()
    assert (state1 == state2).all()
    assert state1.shape == (2, 2)
    np.testing.assert_almost_equal(state1.trace(),1)

# -- operators for testing purposes --
def spin_z(n: int) -> np.ndarray:
    sz = [n / 2]
    while len(sz) < n + 1:
        sz.append(sz[-1] - 1)
    sz = np.diag(sz)
    return sz

def spin_ladder_up(n: int) -> np.ndarray:
    sp = np.zeros((n + 1, n + 1))
    for jj in range(len(sp) - 1):
        sp[jj][jj + 1] = 0.5 * np.sqrt(
            0.5 * n * (0.5 * n + 1)
            - (0.5 * n - jj) * (0.5 * n - jj - 1))
    return sp

def spin_ladder_down(n: int) -> np.ndarray:
    return spin_ladder_up(n).T

def spin_x(n: int) -> np.ndarray:
    return spin_ladder_up(n) + spin_ladder_down(n)

def spin_y(n: int) -> np.ndarray:
    return 1j*(spin_ladder_up(n) - spin_ladder_down(n))

def spin_operators(n: int) -> list:
    return [np.eye(n+1), spin_x(n), spin_y(n), spin_z(n)]

# ------

def test_gibbs_tempo_bad_input():
    aa, tmp, wc, nu = 0.2, 0.1, 5, 1
    jw = lambda w: 2 * aa * w ** nu / (wc ** (nu - 1))
    correlations  = oqupy.CustomSD(jw, wc, cutoff_type='exponential', temperature=tmp)
    correlations_corr  = oqupy.CustomCorrelations(jw)

    dim = 2
    good_system = oqupy.System(0.5 * spin_x(dim))
    bad_system_value = oqupy.System( 0.5 * spin_x(dim + 1))
    bad_system_type = oqupy.TimeDependentSystem(lambda t: 0.5 * spin_x(dim))

    good_bath = oqupy.Bath(0.5 * spin_z(dim), correlations)
    bad_bath_value = oqupy.Bath(0.5 * spin_z(dim + 1), correlations)
    bad_bath_type = oqupy.Bath(0.5 * spin_z(dim), correlations_corr)

    good_parameters = tempo.GibbsParameters(8, 1.0e-5, name="name", description="desc")
    bad_parameters_type = tempo.TempoParameters(0.1, 1.0e-5, name="name", description="desc")

    tempo.GibbsTempo(good_system, good_bath, good_parameters)

    with pytest.raises(ValueError):
        tempo.GibbsTempo(bad_system_value, good_bath, good_parameters)
    with pytest.raises(ValueError):
        tempo.GibbsTempo(good_system, bad_bath_value, good_parameters)

    with pytest.raises(TypeError):
        tempo.GibbsTempo(bad_system_type, good_bath, good_parameters)
    with pytest.raises(TypeError):
        tempo.GibbsTempo(good_system, bad_bath_type, good_parameters)
    with pytest.raises(TypeError):
        tempo.GibbsTempo(good_system, good_bath, bad_parameters_type)

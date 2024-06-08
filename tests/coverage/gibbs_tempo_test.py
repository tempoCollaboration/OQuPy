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

    gibbs_param = tempo.GibbsParameters(temperature, steps, 1.0e-5, name="rough-A")

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


def test_gibbs_tempo_bad_input():
    aa, tmp, wc, nu = 0.2, 0.1, 5, 1
    jw = lambda w: 2 * aa * w ** nu / (wc ** (nu - 1))
    correlations  = oqupy.CustomSD(jw, wc, cutoff_type='exponential', temperature=tmp)

    dim = 2
    good_system = oqupy.System(0.5 * oqupy.operators.spin_x(dim))
    bad_system_value = oqupy.System( 0.5 * oqupy.operators.spin_x(dim + 1))
    bad_system_type = oqupy.TimeDependentSystem(lambda t: 0.5 * oqupy.operators.spin_x(dim))

    good_bath = oqupy.Bath(0.5 * oqupy.operators.spin_z(dim), correlations)
    bad_bath = oqupy.Bath(0.5 * oqupy.operators.spin_z(dim + 1), correlations)

    good_parameters = tempo.GibbsParameters(tmp,8, 1.0e-5, name="name", description="desc")
    bad_parameters_value = tempo.GibbsParameters(tmp + 1, 8, 1.0e-5, name="name", description="desc")
    bad_parameters_type = tempo.TempoParameters(0.1, 1.0e-5, name="name", description="desc")

    keys = ['system', 'bath', 'parameters']
    good_inputs = [good_system, good_bath, good_parameters]
    bad_types = [bad_system_type, 'x', bad_parameters_type]
    bad_values = [bad_system_value, bad_bath, bad_parameters_value]

    for k, t, v in zip(keys, bad_types, bad_values):
        input = dict(zip(keys, good_inputs))
        input[k] = t
        with pytest.raises(TypeError):
            gibbs_sys = tempo.GibbsTempo(**input)
        input[k] = v
        with pytest.raises(ValueError):
            gibbs_sys = tempo.GibbsTempo(**input)


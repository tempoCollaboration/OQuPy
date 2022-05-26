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
Tests for the time_evovling_mpo.tempo module.
"""

import pytest
import numpy as np

import oqupy as tempo

def test_tempo_parameters():
    tempo_param = tempo.TempoParameters(
        0.1, None, 1.0e-5, None, "rough", "bla")
    str(tempo_param)
    assert tempo_param.dt == 0.1
    assert tempo_param.dkmax == None
    assert tempo_param.epsrel == 1.0e-5
    tempo_param.dt = 0.05
    tempo_param.dkmax = 42
    tempo_param.epsrel = 1.0e-6
    assert tempo_param.dt == 0.05
    assert tempo_param.dkmax == 42
    assert tempo_param.epsrel == 1.0e-6
    del tempo_param.dkmax
    assert tempo_param.dkmax == None

def test_tempo_parameters_bad_input():
    with pytest.raises(AssertionError):
        tempo.TempoParameters("x", 42, 1.0e-5, None, "rough", "bla")
    with pytest.raises(AssertionError):
        tempo.TempoParameters(0.1, "x", 1.0e-5, None, "rough", "bla")
    with pytest.raises(AssertionError):
        tempo.TempoParameters(0.1, 42, "x", None, "rough", "bla")

def test_tempo():
    start_time = -0.3
    end_time1 = 0.4
    end_time2 = 0.6
    end_time3 = 0.84

    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    tempo_param_A = tempo.TempoParameters(0.1, 5, 1.0e-5, name="rough-A")
    tempo_sys_A = tempo.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time)
    assert tempo_sys_A.dimension == 2
    tempo_sys_A.compute(end_time=end_time1, progress_type="bar")
    tempo_sys_A.compute(end_time=end_time2, progress_type="silent")
    tempo_sys_A.compute(end_time=end_time3, progress_type="simple")
    dyn_A = tempo_sys_A.get_dynamics()
    assert len(dyn_A.times) == 12

def test_tempo_bad_input():
    start_time = -0.3
    end_time = 0.84

    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    tempo_param_A = tempo.TempoParameters(0.1, 5, 1.0e-5, name="rough-A")
    with pytest.raises(AssertionError):
        tempo_sys_A = tempo.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state="bla",
                                  start_time=start_time)
    with pytest.raises(AssertionError):
        tempo_sys_A = tempo.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state=initial_state,
                                  start_time="bla")

    tempo_sys_A = tempo.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time)
    with pytest.raises(AssertionError):
        tempo_sys_A.compute(end_time="bla", progress_type="bar")


def test_guess_tempo_parameters():
    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(t)+1j*np.sin(6.0*t)) * np.exp(-2.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    with pytest.warns(UserWarning):
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0)

    with pytest.raises(AssertionError): # bad start time input
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time="bla",
                                             end_time=15.0)
    with pytest.raises(AssertionError): # bad end time input
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time="bla")
    with pytest.raises(ValueError): # bad start/end time (swapped)
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=10.0,
                                             end_time=0.0)
    with pytest.raises(AssertionError): # bad tolerance
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance="bla")
    with pytest.raises(AssertionError): # bad tolerance (negative)
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=-0.2)
    with pytest.warns(UserWarning): # reach MAX_DKMAX
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=1.0e-12)

def test_tempo_time_dependent():
    # Tempo class must be able to handle TimeDependentSystem as well
    system = tempo.TimeDependentSystem(lambda t: t * 0.5 * tempo.operators.sigma("z"))
    correlations = tempo.PowerLawSD(alpha=0.2,
                                    zeta=2,
                                    cutoff=1.0,
                                    cutoff_type='exponential')
    bath = tempo.Bath(0.1 * tempo.operators.sigma("x"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=0.1, dkmax=10, epsrel=10**(-2))
    tempo_A= tempo.Tempo(system=system,
                        bath=bath,
                        parameters=tempo_parameters,
                        initial_state=tempo.operators.spin_dm("down"),
                        start_time=0.5)
    dyn_A = tempo_A.compute(0.6)

def test_tempo_compute():
    start_time = -0.3
    end_time = 0.84

    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    with pytest.warns(UserWarning):
        dyn_A = tempo.tempo_compute(system=system,
                                    bath=bath,
                                    initial_state=initial_state,
                                    start_time=start_time,
                                    end_time=end_time)

def test_tempo_dynamics_reference():
    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlations = tempo.PowerLawSD(alpha=0.1,
                                    zeta=1,
                                    cutoff=1.0,
                                    cutoff_type='exponential')
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=0.1, dkmax=10, epsrel=10**(-4))
    tempo_A= tempo.Tempo(system=system,
                        bath=bath,
                        parameters=tempo_parameters,
                        initial_state=tempo.operators.spin_dm("up"),
                        start_time=0.0)
    dynamics_1 = tempo_A.compute(end_time=0.2)
    t_1, sz_1 = dynamics_1.expectations(tempo.operators.sigma("z"))
    tempo_A.compute(end_time=0.4)
    dynamics_2 = tempo_A.get_dynamics()
    t_2, sz_2 = dynamics_2.expectations(tempo.operators.sigma("z"))
    assert dynamics_1 == dynamics_2
    assert len(t_2) > len(t_1)

def test_tempo_with_field():
	system = tempo.TimeDependentSystemWithField(
				lambda t, field: 0.2 * tempo.operators.sigma("x") * np.abs(field),
				lambda t, state, field: -0.1*field -0.1j * np.matmul(
				tempo.operators.sigma("x"), state).trace().real)
	correlations = tempo.PowerLawSD(alpha=0.1,
								zeta=1,
								cutoff=5.0,
								cutoff_type='gaussian',
								temperature=0.1)
	bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
	tempo_parameters = tempo.TempoParameters(dt=0.1, dkmax=20, epsrel=10**(-7))
	tempo_sys = tempo.TempoWithField(system=system,
						bath=bath,
						initial_state=tempo.operators.spin_dm("z-"),
						initial_field=1.0+1.0j,
						start_time=0.0,
						parameters=tempo_parameters)
	dynamics = tempo_sys.compute(end_time=0.5, progress_type="silent")
	assert tempo_sys.dimension == 2
	assert isinstance(dynamics, tempo.dynamics.DynamicsWithField)
	assert len(dynamics.times == 6)
	# bad input
	with pytest.raises(AssertionError):
		# wrong system type
		wrong_system = tempo.TimeDependentSystem(
			lambda t: 0.2 * tempo.operators.sigma("x")
			)
		tempo.TempoWithField(system=wrong_system,
						bath=bath,
						initial_state=tempo.operators.spin_dm("z-"),
						initial_field=1.0+1.0j,
						start_time=0.0,
						parameters=tempo_parameters)
	with pytest.raises(TypeError):
		# no initial field
		tempo.TempoWithField(system=system,
						bath=bath,
						initial_state=tempo.operators.spin_dm("z-"),
						start_time=0.0,
						parameters=tempo_parameters)

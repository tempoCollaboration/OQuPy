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

import oqupy as tempo

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

    tempo_param_A = tempo.TempoParameters(0.1, 1.0e-5, None, 5, name="rough-A")
    # With degeneracy checks off
    tempo_sys_A = tempo.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time,
                              unique=False)
    assert tempo_sys_A.dimension == 2
    tempo_sys_A.compute(end_time=end_time1, progress_type="bar")
    tempo_sys_A.compute(end_time=end_time2, progress_type="silent")
    tempo_sys_A.compute(end_time=end_time3, progress_type="simple")
    dyn_A = tempo_sys_A.get_dynamics()
    assert len(dyn_A.times) == 12
    
    # With degeneracy checks on
    tempo_sys_B = tempo.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time,
                              unique=True)
    assert tempo_sys_B.dimension == 2
    tempo_sys_B.compute(end_time=end_time1, progress_type="bar")
    tempo_sys_B.compute(end_time=end_time2, progress_type="silent")
    tempo_sys_B.compute(end_time=end_time3, progress_type="simple")
    dyn_B = tempo_sys_B.get_dynamics()
    assert len(dyn_B.times) == 12

def test_tempo_bad_input():
    start_time = -0.3
    end_time = 0.84

    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    tempo_param_A = tempo.TempoParameters(0.1, 1.0e-5, None, 5, name="rough-A")
    with pytest.raises(TypeError):
        tempo_sys_A = tempo.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state="bla",
                                  start_time=start_time)
    with pytest.raises(TypeError):
        tempo_sys_A = tempo.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state=initial_state,
                                  start_time="bla")
    
    with pytest.raises(AssertionError):
        tempo_sys_A = tempo.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state=initial_state,
                                  start_time=start_time,
                                  unique="bla")

    tempo_sys_A = tempo.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time)
    with pytest.raises(TypeError):
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

    with pytest.raises(TypeError): # bad start time input
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time="bla",
                                             end_time=15.0)
    with pytest.raises(TypeError): # bad end time input
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time="bla")
    with pytest.raises(ValueError): # bad start/end time (swapped)
        param = tempo.guess_tempo_parameters(bath=bath,
                                             start_time=10.0,
                                             end_time=0.0)
    with pytest.raises(TypeError): # bad tolerance
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
    unique = False

    with pytest.warns(UserWarning):
        dyn_A = tempo.tempo_compute(system=system,
                                    bath=bath,
                                    initial_state=initial_state,
                                    start_time=start_time,
                                    end_time=end_time,
                                    unique=unique)

def test_tempo_dynamics_reference():
    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlations = tempo.PowerLawSD(alpha=0.1,
                                    zeta=1,
                                    cutoff=1.0,
                                    cutoff_type='exponential')
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=0.1, tcut=1.0, epsrel=10**(-4))
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
            lambda t, field: 0.2 * tempo.operators.sigma("x") * np.abs(field))
    field_eom = lambda t, states, field: -0.1*field -0.1j * np.matmul(
            tempo.operators.sigma("x"), states[0]).trace().real
    mean_field_system = tempo.MeanFieldSystem([system], field_eom)
    correlations = tempo.PowerLawSD(alpha=0.1,
                                zeta=1,
                                cutoff=5.0,
                                cutoff_type='gaussian',
                                temperature=0.1)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=0.1, dkmax=20, epsrel=10**(-7))
    tempo_sys = tempo.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[tempo.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=False)
    mean_field_dynamics = tempo_sys.compute(end_time=0.3, progress_type="silent")
    # With degeneracy checks on
    tempo_sysB = tempo.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[tempo.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=True)
    mean_field_dynamicsB = tempo_sysB.compute(end_time=0.3, progress_type="silent")
    assert isinstance(mean_field_dynamics, tempo.dynamics.MeanFieldDynamics)
    assert len(mean_field_dynamics.times == 6)
    # bad input
    wrong_system = tempo.TimeDependentSystemWithField(
            lambda t, a: 0.2 * tempo.operators.sigma("x")
            )
    with pytest.raises(AssertionError):
        # wrong system type
        tempo.MeanFieldTempo(mean_field_system=wrong_system,
                        bath_list=[bath],
                        initial_state_list=[tempo.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=True)
    with pytest.raises(TypeError):
        # no initial field
        tempo.MeanFieldTempo(mean_field_system=[system],
                        bath_list=[bath],
                        initial_state_list=[tempo.operators.spin_dm("z-")],
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=False)
    with pytest.raises(AssertionError):
        # forget lists
        tempo.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=bath,
                        initial_state_list=[tempo.operators.spin_dm("z-")],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # forget lists
        tempo.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=tempo.operators.spin_dm("z-"),
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    big_initial_state = np.eye(3)
    with pytest.raises(ValueError):
        # wrong initial state dimension
        tempo.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[big_initial_state],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    mean_field_system2 = tempo.MeanFieldSystem([system, system],
                                         field_eom)
    with pytest.raises(AssertionError):
        # wrong number of baths
        tempo.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath],
                        initial_state_list=[tempo.operators.sigma('z'),
                          tempo.operators.sigma('z')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # wrong number of initial states
        tempo.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[tempo.operators.sigma('z')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # non-boolean unique option
        tempo.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[tempo.operators.sigma('z'),
                          tempo.operators.sigma('x')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=None)
    with pytest.raises((ValueError, TypeError)):
        # Invalid start time
        tempo.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[tempo.operators.sigma('z'),
                          tempo.operators.sigma('x')],
                        initial_field=1.0,
                        start_time='GO!',
                        parameters=tempo_parameters,
                        unique=False)

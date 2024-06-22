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

def test_tempo():
    start_time = -0.3
    end_time1 = 0.4
    end_time2 = 0.6
    end_time3 = 0.84

    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = oqupy.CustomCorrelations(correlation_function)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    initial_state = oqupy.operators.spin_dm("z+")

    tempo_param_A = oqupy.TempoParameters(0.1, 1.0e-5, None, 5, name="rough-A")
    # With degeneracy checks off
    tempo_sys_A = oqupy.Tempo(system=system,
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
    tempo_sys_B = oqupy.Tempo(system=system,
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
    # With added correlation time and backend config
    tempo_param_C = oqupy.TempoParameters(dt=0.1,
                                          epsrel=1.0e-5, 
                                          dkmax=3,
                                          add_correlation_time=0.5)
    tempo_sys_C = oqupy.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_C,
                              initial_state=initial_state,
                              start_time=0.0,
                              unique=False,
                              backend_config={})
    tempo_sys_C.compute(end_time=0.9)

def test_tempo_bad_input():
    start_time = -0.3
    end_time = 0.84

    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = oqupy.CustomCorrelations(correlation_function)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    initial_state = oqupy.operators.spin_dm("z+")

    tempo_param_A = oqupy.TempoParameters(0.1, 1.0e-5, None, 5, name="rough-A")
    with pytest.raises(TypeError):
        tempo_sys_A = oqupy.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state="bla",
                                  start_time=start_time)
    with pytest.raises(TypeError):
        tempo_sys_A = oqupy.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state=initial_state,
                                  start_time="bla")
    
    with pytest.raises(AssertionError):
        tempo_sys_A = oqupy.Tempo(system=system,
                                  bath=bath,
                                  parameters=tempo_param_A,
                                  initial_state=initial_state,
                                  start_time=start_time,
                                  unique="bla")

    tempo_sys_A = oqupy.Tempo(system=system,
                              bath=bath,
                              parameters=tempo_param_A,
                              initial_state=initial_state,
                              start_time=start_time)
    with pytest.raises(TypeError):
        tempo_sys_A.compute(end_time="bla", progress_type="bar")

def test_tempo_parameters():
    with pytest.raises(AssertionError): # tcut and dkmax
        param = oqupy.TempoParameters(dt=0.1,
                                      tcut=1,
                                      dkmax=10,
                                      epsrel=2.0e-05)
    with pytest.raises(TypeError): # invalid dkmax
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=[],
                                      epsrel=2.0e-05)
    with pytest.raises(ValueError): # invalid dkmax
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=-1,
                                      epsrel=2.0e-05)
    with pytest.raises(TypeError): # invalid tcut
        param = oqupy.TempoParameters(dt=0.1,
                                      tcut='a',
                                      epsrel=2.0e-05)
    with pytest.raises(ValueError): # invalid tcut
        param = oqupy.TempoParameters(dt=0.1,
                                      tcut=-1.0,
                                      epsrel=2.0e-05)
    with pytest.raises(TypeError): # invalid dt
        param = oqupy.TempoParameters(dt=None,
                                      dkmax=10,
                                      epsrel=2.0e-05)
    with pytest.raises(ValueError): # invalid dt
        param = oqupy.TempoParameters(dt=-0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05)
    with pytest.raises(TypeError): # invalid epsrel
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel='a')
    with pytest.raises(ValueError): # invalid epsrel
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=-2.0e-05)
    with pytest.raises(TypeError): # invalid add_correlation
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      add_correlation_time='a')
    with pytest.raises(ValueError): # invalid add_correlation
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      add_correlation_time=-1.0)
    with pytest.raises(TypeError): # invalid subdiv_limit
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      subdiv_limit='a')
    with pytest.raises(ValueError): # invalid subdiv_limit
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      subdiv_limit=-1.0)
    with pytest.raises(TypeError): # invalid liouvillian_epsrel
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      liouvillian_epsrel='a')
    with pytest.raises(ValueError): # invalid liouvillian_epsrel
        param = oqupy.TempoParameters(dt=0.1,
                                      dkmax=10,
                                      epsrel=2.0e-05,
                                      liouvillian_epsrel=-1.0)
    # Cover None sub_div limit, printing parameters
    param = oqupy.TempoParameters(dt=0.1,
                                  dkmax=10,
                                  epsrel=2.0e-05,
                                  subdiv_limit=None)
    print(param)
    # Must be true on creation
    assert np.isclose(param.tcut, param.dkmax * param.dt)
    # Test NO memory cutoff specified (allowed)
    param = oqupy.TempoParameters(dt=0.1, epsrel=1.0e-06)

def test_guess_tempo_parameters():
    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))
    t_system = oqupy.TimeDependentSystem(
            lambda t: t * np.sin(10*t) * oqupy.operators.sigma("z"))
    f_system = oqupy.TimeDependentSystemWithField(lambda t,a: t+a * np.eye(2))
    correlation_function = lambda t: (np.cos(t)+1j*np.sin(6.0*t)) * np.exp(-2.0*t)
    correlations = oqupy.CustomCorrelations(correlation_function)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    with pytest.warns(UserWarning):
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0)
    with pytest.raises(TypeError): # bad start time input
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time="bla",
                                             end_time=15.0)
    with pytest.raises(TypeError): # bad end time input
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time="bla")
    with pytest.raises(ValueError): # bad start/end time (swapped)
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=10.0,
                                             end_time=0.0)
    with pytest.raises(TypeError): # bad tolerance
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance="bla")
    with pytest.raises(AssertionError): # bad tolerance (negative)
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=-0.2)
    with pytest.warns(UserWarning): # reach MAX_DKMAX
        param = oqupy.guess_tempo_parameters(bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=1.0e-12)
    with pytest.raises(AssertionError): # Not a system
        param = oqupy.guess_tempo_parameters(system=bath,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=15.0)
    with pytest.raises(TypeError): # bad max_samples
        param = oqupy.guess_tempo_parameters(system=system,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=0.01,
                                             max_samples='A')
    with pytest.raises(AssertionError): # bad max_samples
        param = oqupy.guess_tempo_parameters(system=system,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=0.01,
                                             max_samples=0)
    with pytest.warns(UserWarning): # reach max_samples
        param = oqupy.guess_tempo_parameters(system=t_system,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=15.0,
                                             tolerance=1.0e-3,
                                             max_samples=30)
    with pytest.warns(UserWarning): # warn about field guess
        param = oqupy.guess_tempo_parameters(system=f_system,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=5.0)
    with pytest.warns(UserWarning):
        # Large system frequencies
        t_system = oqupy.TimeDependentSystem(
            lambda t: 1e2 * t * oqupy.operators.sigma("z"))
        param = oqupy.guess_tempo_parameters(system=t_system,
                                             bath=bath,
                                             start_time=0.0,
                                             end_time=5.0)
    base_system = oqupy.system.BaseSystem(2)
    times = np.array([0.0, 0.1])
    with pytest.raises(TypeError):
        oqupy.tempo._max_tdependentsystem_frequency(base_system, times)

def test_tempo_time_dependent():
    # Tempo class must be able to handle TimeDependentSystem as well
    system = oqupy.TimeDependentSystem(lambda t: t * 0.5 * oqupy.operators.sigma("z"))
    correlations = oqupy.PowerLawSD(alpha=0.2,
                                    zeta=2,
                                    cutoff=1.0,
                                    cutoff_type='exponential')
    bath = oqupy.Bath(0.1 * oqupy.operators.sigma("x"), correlations)
    tempo_parameters = oqupy.TempoParameters(dt=0.1, dkmax=10, epsrel=10**(-2))
    tempo_A= oqupy.Tempo(system=system,
                        bath=bath,
                        parameters=tempo_parameters,
                        initial_state=oqupy.operators.spin_dm("down"),
                        start_time=0.5)
    dyn_A = tempo_A.compute(0.6)

def test_tempo_compute():
    start_time = -0.3
    end_time = 0.84

    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = oqupy.CustomCorrelations(correlation_function)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    initial_state = oqupy.operators.spin_dm("z+")
    unique = False

    with pytest.warns(UserWarning):
        dyn_A = oqupy.tempo_compute(system=system,
                                    bath=bath,
                                    initial_state=initial_state,
                                    start_time=start_time,
                                    end_time=end_time,
                                    unique=unique)

def test_tempo_dynamics_reference():
    system = oqupy.System(0.5 * oqupy.operators.sigma("x"))
    correlations = oqupy.PowerLawSD(alpha=0.1,
                                    zeta=1,
                                    cutoff=1.0,
                                    cutoff_type='exponential')
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    tempo_parameters = oqupy.TempoParameters(dt=0.1, tcut=1.0, epsrel=10**(-4))
    tempo_A= oqupy.Tempo(system=system,
                        bath=bath,
                        parameters=tempo_parameters,
                        initial_state=oqupy.operators.spin_dm("up"),
                        start_time=0.0)
    dynamics_1 = tempo_A.compute(end_time=0.2)
    t_1, sz_1 = dynamics_1.expectations(oqupy.operators.sigma("z"))
    tempo_A.compute(end_time=0.4)
    dynamics_2 = tempo_A.get_dynamics()
    t_2, sz_2 = dynamics_2.expectations(oqupy.operators.sigma("z"))
    assert dynamics_1 == dynamics_2
    assert len(t_2) > len(t_1)

def test_tempo_with_field():
    system = oqupy.TimeDependentSystemWithField(
            lambda t, field: 0.2 * oqupy.operators.sigma("x") * np.abs(field))
    field_eom = lambda t, states, field: -0.1*field -0.1j * np.matmul(
            oqupy.operators.sigma("x"), states[0]).trace().real
    mean_field_system = oqupy.MeanFieldSystem([system], field_eom)
    correlations = oqupy.PowerLawSD(alpha=0.1,
                                zeta=1,
                                cutoff=5.0,
                                cutoff_type='gaussian',
                                temperature=0.1)
    bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
    tempo_parameters = oqupy.TempoParameters(dt=0.1, dkmax=20, epsrel=10**(-7))
    tempo_sys = oqupy.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[oqupy.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=False)
    mean_field_dynamics = tempo_sys.compute(end_time=0.3, progress_type="silent")
    # With degeneracy checks on, plus backend config (empty)
    tempo_sysB = oqupy.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[oqupy.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=True,
                        backend_config={})
    mean_field_dynamicsB = tempo_sysB.compute(end_time=0.3, progress_type="silent")
    mean_field_dynamicsB_get = tempo_sysB.get_dynamics()
    assert mean_field_dynamicsB == mean_field_dynamicsB_get
    assert isinstance(mean_field_dynamics, oqupy.dynamics.MeanFieldDynamics)
    assert len(mean_field_dynamics.times == 6)
    # bad input
    wrong_system = oqupy.TimeDependentSystemWithField(
            lambda t, a: 0.2 * oqupy.operators.sigma("x")
            )
    with pytest.raises(AssertionError):
        # wrong system type
        oqupy.MeanFieldTempo(mean_field_system=wrong_system,
                        bath_list=[bath],
                        initial_state_list=[oqupy.operators.spin_dm("z-")],
                        initial_field=1.0+1.0j,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=True)
    with pytest.raises(TypeError):
        # no initial field
        oqupy.MeanFieldTempo(mean_field_system=[system],
                        bath_list=[bath],
                        initial_state_list=[oqupy.operators.spin_dm("z-")],
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=False)
    with pytest.raises(AssertionError):
        # forget lists
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=bath,
                        initial_state_list=[oqupy.operators.spin_dm("z-")],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # forget lists
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=oqupy.operators.spin_dm("z-"),
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    big_initial_state = np.eye(3)
    with pytest.raises(ValueError):
        # wrong initial state dimension
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system,
                        bath_list=[bath],
                        initial_state_list=[big_initial_state],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    mean_field_system2 = oqupy.MeanFieldSystem([system, system],
                                         field_eom)
    with pytest.raises(AssertionError):
        # wrong number of baths
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath],
                        initial_state_list=[oqupy.operators.sigma('z'),
                          oqupy.operators.sigma('z')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # wrong number of initial states
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[oqupy.operators.sigma('z')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters)
    with pytest.raises(AssertionError):
        # non-boolean unique option
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[oqupy.operators.sigma('z'),
                          oqupy.operators.sigma('x')],
                        initial_field=1.0,
                        start_time=0.0,
                        parameters=tempo_parameters,
                        unique=None)
    with pytest.raises((ValueError, TypeError)):
        # Invalid start time
        oqupy.MeanFieldTempo(mean_field_system=mean_field_system2,
                        bath_list=[bath, bath],
                        initial_state_list=[oqupy.operators.sigma('z'),
                          oqupy.operators.sigma('x')],
                        initial_field=1.0,
                        start_time='GO!',
                        parameters=tempo_parameters,
                        unique=False)

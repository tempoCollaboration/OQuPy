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
Tests for the time_evovling_mpo.pt_tempo module.
"""

import pytest
import numpy as np

import oqupy as tempo

def test_pt_tempo_parameters():
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

def test_pt_tempo_parameters_bad_input():
    with pytest.raises(AssertionError):
        tempo.TempoParameters("x", 42, 1.0e-5, None, "rough", "bla")
    with pytest.raises(AssertionError):
        tempo.TempoParameters(0.1, "x", 1.0e-5, None, "rough", "bla")
    with pytest.raises(AssertionError):
        tempo.TempoParameters(0.1, 42, "x", None, "rough", "bla")

def test_pt_tempo():
    start_time = -0.3
    end_time = 0.84

    system = tempo.System(0.5 * tempo.operators.sigma("x"))
    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    tempo_param_A = tempo.TempoParameters(0.1, 5, 1.0e-5, name="rough-A")
    pt_tempo_sys_A = tempo.PtTempo(bath=bath,
                                   parameters=tempo_param_A,
                                   start_time=start_time,
                                   end_time=end_time)
    assert pt_tempo_sys_A.dimension == 2
    pt_tempo_sys_A.compute()
    pt_A = pt_tempo_sys_A.get_process_tensor()
    assert len(pt_A) == 11

    tempo_param_B = tempo.TempoParameters(0.1, None, 1.0e-5, name="rough-B")
    pt_tempo_sys_B = tempo.PtTempo(bath=bath,
                                   parameters=tempo_param_B,
                                   start_time=start_time,
                                   end_time=end_time)
    pt_B = pt_tempo_sys_B.get_process_tensor()


def test_tempo_bad_input():
    start_time = -0.3
    end_time = 0.84

    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)

    tempo_param = tempo.TempoParameters(0.1, 5, 1.0e-5, name="rough-A")

    tempo.PtTempo(bath=bath,
                  parameters=tempo_param,
                  start_time=start_time,
                  end_time=end_time)

    with pytest.raises(AssertionError):
        tempo.PtTempo(bath=bath,
                      parameters=tempo_param,
                      start_time="bla",
                      end_time=end_time)
    with pytest.raises(AssertionError):
        tempo.PtTempo(bath=bath,
                      parameters=tempo_param,
                      start_time=start_time,
                      end_time="bla")
    with pytest.raises(AssertionError):
        tempo.PtTempo(bath=bath,
                      parameters=tempo_param,
                      start_time=0.0,
                      end_time=0.0001)

def test_tempo_compute():
    start_time = -0.3
    end_time = 0.84

    correlation_function = lambda t: (np.cos(6.0*t)+1j*np.sin(6.0*t)) \
                                        * np.exp(-12.0*t)
    correlations = tempo.CustomCorrelations(correlation_function)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)
    initial_state = tempo.operators.spin_dm("z+")

    with pytest.warns(UserWarning):
        pt_A = tempo.pt_tempo_compute(bath=bath,
                                      start_time=start_time,
                                      end_time=end_time)

def test_compute_dynamics_with_field():
    start_time = 1
    num_steps = 3
    initial_field = 0.5j
    
    system = tempo.TimeDependentSystemWithField(
    			lambda t, field: 1j * 0.2 * tempo.operators.sigma("y") * field,
    			lambda t, state, field: -1j*field -0.1j * np.matmul(
    			tempo.operators.sigma("y"), state).trace().real)
    correlations = tempo.PowerLawSD(alpha=3,
    							zeta=1,
    							cutoff=1.0,
    							cutoff_type='gaussian',
    							temperature=0.0)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("x"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=0.2, dkmax=20, epsrel=10**(-5))
    
    dynamics = tempo.compute_dynamics_with_field(
            system,
            initial_field,
            np.array([[0.5,-0.51j],[0,-.25]]),
            dt = 0.2,
            num_steps = num_steps,
            start_time = start_time
            )
    
    assert isinstance(dynamics, tempo.dynamics.DynamicsWithField) # check correct type of dynamics returned
    assert len(dynamics.times) == 1 + num_steps # initial time plus num_steps steps 
    assert np.isclose(np.min(dynamics.times), start_time)  # check start_time used correctly
    assert type(dynamics.fields[1]) == tempo.config.NpDtype # check complex type
    assert np.isclose(dynamics.fields[0], initial_field) # check initial field recorded correctly 
    assert len(dynamics.fields) == len(dynamics.times) # check for OBOE in number of fields
    
    # Test subdiv_limit == None
    dynamics2 = tempo.compute_dynamics_with_field(
            system,
            initial_field,
            np.array([[0.5,-0.51j],[0,-.25]]),
            dt = 0.2,
            num_steps = num_steps,
            start_time = start_time,
            subdiv_limit = None
            )
    # Sampling should give same result up to numerical error
    assert np.isclose(dynamics.fields[-1], dynamics2.fields[-1])
    
    # input checks
    # No initial field / wrong type
    with pytest.raises(TypeError):
        tempo.compute_dynamics_with_field(
                system,
                np.eye(2),
                dt = 0.2,
                )
    with pytest.raises(TypeError):
        tempo.compute_dynamics_with_field(
            system,
            None,
            np.eye(2),
            dt = 0.2,
            )
    # Wrong system type
    with pytest.raises(TypeError):
        tempo.compute_dynamics_with_field(
                tempo.TimeDependentSystem(lambda t: 0.5 * t * np.eye(2)),
                initial_field,
                np.array([[0.5,-0.51j],[0,-.25]]),
                dt = 0.2,
                num_steps = num_steps,
                start_time = start_time
                )
    

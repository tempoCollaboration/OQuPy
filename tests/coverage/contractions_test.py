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

def test_compute_dynamics_with_field():
    start_time = 1
    num_steps = 3
    dt = 0.2
    end_time = start_time + (num_steps-1) * dt
    initial_field = 0.5j
    
    system = tempo.TimeDependentSystemWithField(
                lambda t, field: 1j * 0.2 * tempo.operators.sigma("y") * field)
    field_eom = lambda t, states, field: -1j*field -0.1j * np.matmul(
                tempo.operators.sigma("y"), states[0]).trace().real
    mean_field_system = tempo.MeanFieldSystem([system], field_eom)
    correlations = tempo.PowerLawSD(alpha=3,
                                zeta=1,
                                cutoff=1.0,
                                cutoff_type='gaussian',
                                temperature=0.0)
    bath = tempo.Bath(0.5 * tempo.operators.sigma("x"), correlations)
    tempo_parameters = tempo.TempoParameters(dt=dt, dkmax=20, epsrel=10**(-5))
    
    mean_field_dynamics = tempo.compute_dynamics_with_field(
            mean_field_system,
            initial_field,
            initial_state_list = [np.array([[0.5,-0.51j],[0,-.25]])],
            dt = dt,
            num_steps = num_steps,
            start_time = start_time
            )
    
    pt = tempo.pt_tempo_compute(bath=bath,
                                      start_time=0.0,
                                      end_time=end_time,
                                      parameters=tempo_parameters)
    assert isinstance(mean_field_dynamics, tempo.dynamics. MeanFieldDynamics) 
    assert len(mean_field_dynamics.times) == 1 + num_steps 
    assert np.isclose(np.min(mean_field_dynamics.times), start_time)  
    assert type(mean_field_dynamics.fields[1]) == tempo.config.NpDtype 
    assert np.isclose(mean_field_dynamics.fields[0], initial_field) 
    # check correct type of dynamics returned
    assert isinstance(mean_field_dynamics, tempo.dynamics.MeanFieldDynamics)
    # initial time plus num_steps steps 
    assert len(mean_field_dynamics.times) == 1 + num_steps 
    # check start_time used correctly
    assert np.isclose(np.min(mean_field_dynamics.times), start_time)
    # check complex type
    assert type(mean_field_dynamics.fields[1]) == tempo.config.NpDtype
    # check initial field recorded correctly 
    assert np.isclose(mean_field_dynamics.fields[0], initial_field) 
    
    # Test subdiv_limit == None
    mean_field_dynamics2 = tempo.compute_dynamics_with_field(
            mean_field_system,
            initial_field,
            initial_state_list = [np.array([[0.5,-0.51j],[0,-.25]])],
            dt = 0.2,
            num_steps = num_steps,
            start_time = start_time,
            subdiv_limit = None
            )
    
    # input checks
    # No initial field / wrong type
    with pytest.raises(TypeError):
        tempo.compute_dynamics_with_field(
                mean_field_system,
                initial_state_list = [np.eye(2)],
                dt = dt,
                )
    with pytest.raises(TypeError):
        tempo.compute_dynamics_with_field(
            mean_field_system,
            None,
            initial_state_list=[np.eye(2)],
            dt = dt,
            )
    # Wrong system type
    with pytest.raises(AssertionError):
        tempo.compute_dynamics_with_field(
                tempo.TimeDependentSystem(lambda t: 0.5 * t * np.eye(2)),
                initial_field,
                initial_state_list = [np.array([[0.5,-0.51j],[0,-.25]])],
                dt = dt,
                num_steps = num_steps,
                start_time = start_time
                )
    # too many process tensors
    with pytest.raises(AssertionError):
        tempo.compute_dynamics_with_field(
                mean_field_system,
                initial_field,
                initial_state_list = [np.array([[0.5,-0.51j],[0,-.25]])],
                process_tensor_list = [pt, pt],
                dt = dt,
                num_steps = num_steps,
                start_time = start_time
                )
    # forget lists
    with pytest.raises(AssertionError):
        tempo.compute_dynamics_with_field(
                mean_field_system,
                initial_field,
                initial_state_list = [np.array([[0.5,-0.51j],[0,-.25]])],
                process_tensor_list = pt,
                dt = dt,
                num_steps = num_steps,
                start_time = start_time
                )
    with pytest.raises(AssertionError):
        tempo.compute_dynamics_with_field(
                mean_field_system,
                initial_field,
                initial_state_list = np.array([[0.5,-0.51j],[0,-.25]]),
                process_tensor_list = [pt],
                dt = dt,
                num_steps = num_steps,
                start_time = start_time
                )
    #too many steps for process tensor
    with pytest.raises(AssertionError):
        tempo.compute_dynamics_with_field(
                mean_field_system,
                initial_field,
                initial_state_list = np.array([[0.5,-0.51j],[0,-.25]]),
                process_tensor_list = [pt],
                dt = dt,
                num_steps = 2*num_steps,
                start_time = start_time
                )

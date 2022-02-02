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
Tests for the time_evovling_mpo.backends.tensor_network modules.
"""

import pytest
import numpy as np

import oqupy
from oqupy import process_tensor

# -----------------------------------------------------------------------------
# -- Test C: Spin-1 boson model -------------------------------------------------

# Initial state:
initial_state_C = np.array([[0.0,0.0,0.0],
                            [0.0,1.0,0.0],
                            [0.0,0.0,0.0]])

# System operator
h_sys_C = np.array([[1.0,0.0,0.0],
                    [0.0,0.5,0.0],
                    [0.0,0.0,0.0]])
h_sys_C += np.array([[0.0,0.5,0.0],
                     [0.5,0.0,0.5],
                     [0.0,0.5,0.0]])

# Markovian dissipation
gamma_C_1 = 0.1
gamma_C_2 = 0.2
lindblad_operators_C_1 = np.array([[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
lindblad_operators_C_2 = np.array([[0.5, 0.0, 0.0],
                                   [0.0,-0.5, 0.0],
                                   [0.0, 0.0, 0.0]])

# Ohmic spectral density with exponential cutoff
coupling_operator_C = np.array([[0.5, 0.0, 0.0],
                                [0.0,-0.5, 0.0],
                                [0.0, 0.0, 0.0]])
alpha_C = 0.3
cutoff_C = 5.0
temperature_C = 0.0

# end time
t_end_C = 1.0

# result obtained with release code (made hermitian, dkmax=10):
rho_C = np.array(
    [[ 0.12576653+0.j        ,-0.11739956-0.14312036j, 0.12211454-0.05963583j],
     [-0.11739956+0.14312036j, 0.61315893+0.j        ,-0.06636825+0.26917271j],
     [ 0.12211454+0.05963583j,-0.06636825-0.26917271j, 0.26107455+0.j        ]])

correlations_C = oqupy.PowerLawSD(alpha=alpha_C,
                                  zeta=1.0,
                                  cutoff=cutoff_C,
                                  cutoff_type="exponential",
                                  temperature=temperature_C,
                                  name="ohmic")
bath_C = oqupy.Bath(coupling_operator_C,
                    correlations_C,
                    name="phonon bath")
system_C = oqupy.System(h_sys_C,
                        gammas=[gamma_C_1, gamma_C_2],
                        lindblad_operators=[lindblad_operators_C_1,
                                            lindblad_operators_C_2])


# -----------------------------------------------------------------------------

def test_tensor_network_tempo_backend_C():
    tempo_params_C = oqupy.TempoParameters(
        dt=0.05,
        dkmax=10,
        epsrel=10**(-7),
        add_correlation_time=None)
    tempo_C = oqupy.Tempo(system_C,
                          bath_C,
                          tempo_params_C,
                          initial_state_C,
                          start_time=0.0)
    tempo_C.compute(end_time=1.0)
    dyn_C = tempo_C.get_dynamics()
    assert dyn_C.times[-1] == 1.0
    np.testing.assert_almost_equal(dyn_C.states[-1], rho_C, decimal=4)


def test_tensor_network_pt_tempo_backend_C():
    tempo_params_C = oqupy.TempoParameters(
        dt=0.05,
        dkmax=10,
        epsrel=10**(-7),
        add_correlation_time=None)
    pt = oqupy.pt_tempo_compute(
                bath_C,
                start_time=0.0,
                end_time=1.0,
                parameters=tempo_params_C)

    dyn = oqupy.compute_dynamics(
        system=system_C,
        process_tensor=pt,
        initial_state=initial_state_C)
    assert dyn.times[-1] == 1.0
    np.testing.assert_almost_equal(dyn.states[-1], rho_C, decimal=4)

# -----------------------------------------------------------------------------

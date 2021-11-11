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

import oqupy as tempo

# -----------------------------------------------------------------------------
# -- Test A: Spin boson model -------------------------------------------------

# Initial state:
initial_state_D = np.array([[1.0,0.0],[0.0,0.0]])

# System operator
h_sys_D = lambda t:  t*np.array([[0.0,0.5],[0.5,0.0]])

# Markovian dissipation
gamma_D_1 = lambda t: t*0.1 # with sigma minus
gamma_D_2 = lambda t: t*0.2 # with sigma z

# Ohmic spectral density with exponential cutoff
coupling_operator_D = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_D = 0.3
cutoff_D = 5.0
temperature_D = 0.2

# end time
t_end_D = 1.0

# result obtained with release code (made hermitian):
rho_D = np.array([[ 0.91244238 +0.0j           , -0.0586503  +1.33156274e-01j],
                  [-0.0586503  -1.33156274e-01j,  0.08755133 +0.0j]])

correlations_D = tempo.PowerLawSD(alpha=alpha_D,
                                  zeta=1.0,
                                  cutoff=cutoff_D,
                                  cutoff_type="exponential",
                                  temperature=temperature_D,
                                  name="ohmic")
bath_D = tempo.Bath(coupling_operator_D,
                    correlations_D,
                    name="phonon bath")
system_D = tempo.TimeDependentSystem(h_sys_D,
                        gammas=[gamma_D_1, gamma_D_2],
                        lindblad_operators=[
                            lambda t: tempo.operators.sigma("-"),
                            lambda t: tempo.operators.sigma("z")])

# -----------------------------------------------------------------------------

@pytest.mark.parametrize('backend,backend_config',
                        [("tensor-network", {"backend":"numpy"}),
                         ("tensor-network", None)])

def test_tensor_network_tempo_backend_D(backend, backend_config):
    tempo_params_D = tempo.TempoParameters(dt=0.05,
                                           dkmax=None,
                                           epsrel=10**(-7))
    tempo_D = tempo.Tempo(system_D,
                          bath_D,
                          tempo_params_D,
                          initial_state_D,
                          start_time=0.0,
                          backend=backend,
                          backend_config=backend_config)
    tempo_D.compute(end_time=1.0)
    dyn_D = tempo_D.get_dynamics()
    print(dyn_D.states[-1])
    np.testing.assert_almost_equal(dyn_D.states[-1], rho_D, decimal=4)

@pytest.mark.parametrize('backend,backend_config',
                        [("tensor-network", {"backend":"numpy"}),
                         ("tensor-network", None)])
def test_tensor_network_pt_tempo_backend_D(backend,backend_config):
    tempo_params_D = tempo.PtTempoParameters(dt=0.05,
                                             dkmax=None,
                                             epsrel=10**(-7))
    pt = tempo.pt_tempo_compute(
                bath_D,
                start_time=0.0,
                end_time=1.0,
                parameters=tempo_params_D,
                backend=backend,
                backend_config=backend_config)
    state = pt.compute_final_state_from_system(system=system_D,
                                               initial_state=initial_state_D)
    np.testing.assert_almost_equal(state, rho_D, decimal=4)

    dyn = pt.compute_dynamics_from_system(system=system_D,
                                          initial_state=initial_state_D)
    print(dyn.states[-1])
    np.testing.assert_almost_equal(dyn.states[-1], rho_D, decimal=4)
# -----------------------------------------------------------------------------

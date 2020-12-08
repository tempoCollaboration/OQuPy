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

import time_evolving_mpo as tempo

# -- Test A: Spin boson model -------------------------------------------------
# Initial state:
initial_state_A = np.array([[1.0,0.0],[0.0,0.0]])
# System operator
h_sys_A = np.array([[0.0,0.5],[0.5,0.0]])
# Markovian dissipation
gamma_A_1 = 0.1 # with sigma minus
gamma_A_2 = 0.2 # with sigma z
# Ohmic spectral density
coupling_operator_A = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_A = 0.3
cutoff_A = 5.0
temperature_A = 0.2
# end time
t_end_A = 1.0
# result obtained with release code (made hermitian):
rho_A = np.array([[ 0.7809559 +0.j        , -0.09456333+0.16671419j],
                  [-0.09456333-0.16671419j,  0.2190441 +0.j        ]])

# -----------------------------------------------------------------------------


def test_tensor_network_tempo_backend_A():
    correlations_A = tempo.PowerLawSD(alpha=alpha_A,
                                      zeta=1.0,
                                      cutoff=cutoff_A,
                                      cutoff_type="exponential",
                                      temperature=temperature_A,
                                      name="ohmic")
    bath_A = tempo.Bath(coupling_operator_A,
                        correlations_A,
                        name="phonon bath")
    system_A = tempo.System(h_sys_A,
                            gammas=[gamma_A_1, gamma_A_2],
                            lindblad_operators=[tempo.operators.sigma("-"),
                                                tempo.operators.sigma("z")])
    tempo_params_A = tempo.TempoParameters(dt=0.05,
                                           dkmax=None,
                                           epsrel=10**(-7))
    tempo_A = tempo.Tempo(system_A,
                          bath_A,
                          tempo_params_A,
                          initial_state_A,
                          start_time=0.0,
                          backend_name="tensor-network")
    tempo_A.compute(end_time=1.0)
    dyn_A = tempo_A.get_dynamics()
    np.testing.assert_almost_equal(dyn_A.states[-1], rho_A, decimal=4)

def test_tensor_network_tempo_backend_non_diag():
    Omega = 1.0
    omega_cutoff = 5.0
    alpha = 0.3

    sx=tempo.operators.sigma("x")
    sy=tempo.operators.sigma("y")
    sz=tempo.operators.sigma("z")

    bases = [{"sys_op":sx, "coupling_op":sz, \
                "init_state":tempo.operators.spin_dm("plus-y")},
             {"sys_op":sy, "coupling_op":sx, \
                "init_state":tempo.operators.spin_dm("up")},
             {"sys_op":sz, "coupling_op":sy, \
                "init_state":tempo.operators.spin_dm("plus")}]

    results = []
    for i, base in enumerate(bases):
        system = tempo.System(0.5*base["sys_op"])
        correlations = tempo.PowerLawSD(alpha=alpha,
                                        zeta=1,
                                        cutoff=omega_cutoff,
                                        cutoff_type='exponential',
                                        max_correlation_time=8.0)
        bath = tempo.Bath(0.5*base["coupling_op"], correlations)
        tempo_parameters = tempo.TempoParameters(dt=0.1,
                                                 dkmax=30,
                                                 epsrel=10**(-5))

        dynamics = tempo.tempo_compute(system=system,
                                       bath=bath,
                                       initial_state=base["init_state"],
                                       start_time=0.0,
                                       end_time=1.0,
                                       parameters=tempo_parameters)

        _, s_x = dynamics.expectations(0.5*tempo.operators.sigma("x"),
                                       real=True)
        _, s_y = dynamics.expectations(0.5*tempo.operators.sigma("y"),
                                       real=True)
        _, s_z = dynamics.expectations(0.5*tempo.operators.sigma("z"),
                                       real=True)
        if i == 0:
            results.append(np.array([s_x, s_y, s_z]))
        elif i == 1:
            results.append(np.array([s_y, s_z, s_x]))
        elif i == 2:
            results.append(np.array([s_z, s_x, s_y]))

    assert np.allclose(results[0], results[1], atol=tempo_parameters.epsrel)
    assert np.allclose(results[0], results[2], atol=tempo_parameters.epsrel)

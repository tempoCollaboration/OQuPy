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
Tests for the oqupy.tempo modules.
"""

import pytest
import numpy as np

import oqupy
from oqupy import process_tensor


# -----------------------------------------------------------------------------
# -- Test A: Spin boson model -------------------------------------------------

# Initial state:
initial_state_A = np.array([[1.0,0.0],[0.0,0.0]])

# System operator
h_sys_A = np.array([[0.0,0.5],[0.5,0.0]])

# Markovian dissipation
gamma_A_1 = 0.1 # with sigma minus
gamma_A_2 = 0.2 # with sigma z

# Ohmic spectral density with exponential cutoff
coupling_operator_A = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_A = 0.3
cutoff_A = 5.0
temperature_A = 0.2

# end time
t_end_A = 1.0

# result obtained with release code (made hermitian):
rho_A = np.array([[ 0.7809559 +0.j        , -0.09456333+0.16671419j],
                  [-0.09456333-0.16671419j,  0.2190441 +0.j        ]])

correlations_A = oqupy.PowerLawSD(alpha=alpha_A,
                                  zeta=1.0,
                                  cutoff=cutoff_A,
                                  cutoff_type="exponential",
                                  temperature=temperature_A,
                                  name="ohmic")
bath_A = oqupy.Bath(coupling_operator_A,
                    correlations_A,
                    name="phonon bath")
system_A = oqupy.System(h_sys_A,
                        gammas=[gamma_A_1, gamma_A_2],
                        lindblad_operators=[oqupy.operators.sigma("-"),
                                            oqupy.operators.sigma("z")])

# -----------------------------------------------------------------------------
# -- Test B: Collective Ising chain -------------------------------------------

# Initial state:
initial_state_B = np.array([[1.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]])

# System operator
j_value = -1.0
h = 1.0
s_z = np.array([[1.0,0.0,0.0],
                [0.0,0.0,0.0],
                [0.0,0.0,-1.0]])
s_x = np.array([[0.0,1.0,0.0],
                [1.0,0.0,1.0],
                [0.0,1.0,0.0]])
h_sys_B = (j_value/2) * s_z @ s_z + h * s_x

# Ohmic spectral density with exponential cutoff
coupling_operator_B = s_z
alpha_B = 0.3
cutoff_B = 5.0
temperature_B = 0.2

# end time
t_end_B = 1.0

correlations_B = oqupy.PowerLawSD(alpha=alpha_B,
                                  zeta=1.0,
                                  cutoff=cutoff_B,
                                  cutoff_type="exponential",
                                  temperature=temperature_B,
                                  name="ohmic")
bath_B = oqupy.Bath(coupling_operator_B,
                    correlations_B,
                    name="phonon bath")
system_B = oqupy.System(h_sys_B)


# -----------------------------------------------------------------------------
# -- Test C: 1D bath coupling -------------------------------------------------

# Initial state:
initial_state_C = np.array([[1.0,0.0,0.0],
                            [0.0,0.0,0.0],
                            [0.0,0.0,0.0]])

# System operator
h_sys_C = -2 * s_z @ s_z + s_x

# Ohmic spectral density with exponential cutoff
coupling_operator_C = np.eye(3)
alpha_C = 0.3
cutoff_C = 5.0
temperature_C = 0.0

# end time
t_end_C = 2.0

correlations_C = oqupy.PowerLawSD(alpha=alpha_C,
                                  zeta=1.0,
                                  cutoff=cutoff_C,
                                  cutoff_type="exponential",
                                  temperature=temperature_C,
                                  name="ohmic")
bath_C = oqupy.Bath(coupling_operator_C,
                    correlations_C,
                    name="phonon bath")
system_C = oqupy.System(h_sys_C)


# -----------------------------------------------------------------------------

def test_degeneracy_exact():
    tempo_params_A = oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))
    tempo_A = oqupy.Tempo(
        system_A,
        bath_A,
        tempo_params_A,
        initial_state_A,
        start_time=0.0,
        unique=True)
    tempo_A.compute(end_time=t_end_A)
    dyn_A = tempo_A.get_dynamics()
    np.testing.assert_almost_equal(dyn_A.states[-1], rho_A, decimal=4)

def test_degeneracy_compare():
    tempo_params_B = oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))
    tempo_unique = oqupy.Tempo(
        system_B,
        bath_B,
        tempo_params_B,
        initial_state_B,
        start_time=0.0,
        unique=True)
    tempo_non_unique = oqupy.Tempo(
        system_B,
        bath_B,
        tempo_params_B,
        initial_state_B,
        start_time=0.0,
        unique=False)
    tempo_unique.compute(end_time=t_end_B)
    tempo_non_unique.compute(end_time=t_end_B)
    dyn_unique = tempo_unique.get_dynamics()
    dyn_non_unique = tempo_non_unique.get_dynamics()
    np.testing.assert_almost_equal(dyn_unique.states, dyn_non_unique.states,
                                   decimal=4)

def test_degeneracy_1d():
    tempo_params_C = oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))
    tempo_unique = oqupy.Tempo(
        system_C,
        bath_C,
        tempo_params_C,
        initial_state_C,
        start_time=0.0,
        unique=True)
    tempo_non_unique = oqupy.Tempo(
        system_C,
        bath_C,
        tempo_params_C,
        initial_state_C,
        start_time=0.0,
        unique=False)
    tempo_unique.compute(end_time=t_end_C)
    tempo_non_unique.compute(end_time=t_end_C)
    dyn_unique = tempo_unique.get_dynamics()
    dyn_non_unique = tempo_non_unique.get_dynamics()
    np.testing.assert_almost_equal(dyn_unique.states, dyn_non_unique.states,
                                   decimal=4)

# -----------------------------------------------------------------------------

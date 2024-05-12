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
from oqupy import tempo
from oqupy import process_tensor


# -----------------------------------------------------------------------------
# -- Test A: Zero coupling -------------------------------------------------
# -- Test B: Independent boson model -------------------------------------------------


# System operator
h_sys_A = np.array([[0.5, 0.0], [0, -0.5]])


# Ohmic spectral density with exponential cutoff
coupling_operator_A = np.array([[1.0, 2.0], [2.0, 4.0]])
coupling_operator_B = np.array([[1.0, 0.0], [0.0, 0.0]])
alpha_A = 0
alpha_B = 0.3
cutoff_A = 5.0
temperature_A = 2.1



rho_A = lambda a: 1 / 2 * (np.eye(2) - 2 * np.tanh(1 / (2 * temperature_A) - a * cutoff_A / temperature_A) * h_sys_A)

rho_B = np.array([[ 0.7809559 +0.j        , -0.09456333+0.16671419j],
                  [-0.09456333-0.16671419j,  0.2190441 +0.j        ]])

correlations_A = oqupy.PowerLawSD(alpha=alpha_A,
                                  zeta=1.0,
                                  cutoff=cutoff_A,
                                  cutoff_type="exponential",
                                  temperature=temperature_A,
                                  name="ohmic")

correlations_B = oqupy.PowerLawSD(alpha=alpha_B,
                                  zeta=1.0,
                                  cutoff=cutoff_A,
                                  cutoff_type="exponential",
                                  temperature=temperature_A,
                                  name="ohmic")


bath_A = oqupy.Bath(coupling_operator_A,
                    correlations_A,
                    name="phonon bath")

bath_B = oqupy.Bath(coupling_operator_B,
                    correlations_B,
                    name="phonon bath")
system_A = oqupy.System(h_sys_A)

gibbs_params_A = tempo.GibbsParameters(
        temperature=temperature_A,
        n_steps=100,
        epsrel=10**(-11))
# -----------------------------------------------------------------------------

def test_gibbs_backend_A():
    tempo_A = tempo.GibbsTempo(
        system_A,
        bath_A,
        gibbs_params_A)
    tempo_A.compute()
    gibbs_A = tempo_A.get_state()
    print('A', gibbs_A)
    print('A', rho_A)
    np.testing.assert_almost_equal(gibbs_A, rho_A(0), decimal=8)

def test_gibbs_backend_B():
    tempo_B = tempo.GibbsTempo(
        system_A,
        bath_B,
        gibbs_params_A)
    tempo_B.compute()
    gibbs_B = tempo_B.get_state()
    print(gibbs_B)
    np.testing.assert_almost_equal(gibbs_B, rho_A(alpha_B), decimal=8)

# -----------------------------------------------------------------------------

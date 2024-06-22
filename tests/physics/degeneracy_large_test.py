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
Tests for the degeneracy checking with TEMPO.
"""

import pytest
import numpy as np

import oqupy
from oqupy import process_tensor


# -----------------------------------------------------------------------------
# -- Test C: Collective Ising Chain with different bath coupling operator

# Initial state:
initial_state_C = np.array([[1.0,0.0,0.0],
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
h_sys_C = (j_value/2) * s_z @ s_z + h * s_x

# Ohmic spectral density with exponential cutoff
coupling_operator_C = np.diag([1,1,2])
alpha_C = 0.3
cutoff_C = 5.0
temperature_C = 0.2

# end time
t_end_C = 5.0

correlations_C = oqupy.PowerLawSD(alpha=alpha_C,
                                  zeta=1.0,
                                  cutoff=cutoff_C,
                                  cutoff_type="exponential",
                                  temperature=temperature_C,
                                  name="ohmic")
bath_C = oqupy.Bath(coupling_operator_C,
                    correlations_C,
                    name="bath with north degeneracies")
system_C = oqupy.System(h_sys_C)

@pytest.mark.skip(reason="See GitHub Issue#115")
def test_large_degeneracy_compare():
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

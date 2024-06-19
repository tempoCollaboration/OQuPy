#Copyright 2020 The TEMPO Collaboration
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
Tests for MeanFieldTempo.
"""
import numpy as np

import oqupy
from oqupy import operators

# -- Test G: Dicke model --

initial_state_G = np.array([[0.5, 0.5], [0.5, 0.5]])
initial_field_G = 1.0

t_end_G = 1.0

# Final state and field obtained from original implementation in tempo-py code
# See tempo-py/src/pfw1/examples/example_G_test.py
rho_G = np.array([[0.6245009+3.19373236e-15j,0.14243496-2.19523032e-01j],
 [0.14243496+2.19523032e-01j,0.3754991 -2.26692588e-15j]])
field_G = 0.10602369935009-0.46986388684474406j


h_sys_G = lambda t, field: 0.5 * operators.sigma("z")  \
        + np.real(field) * operators.sigma("x")
field_eom_G = lambda t, states, field: -(1j + 1) * field \
        - 0.5j * np.matmul(operators.sigma("x"), states[0]).trace().real

correlations_G = oqupy.PowerLawSD(alpha=0.1,
                                  zeta=1.0,
                                  cutoff=5.0,
                                  cutoff_type="gaussian",
                                  temperature=0.0,
                                  name="ohmic")
bath_G = oqupy.Bath(0.5 * operators.sigma("z"),
                    correlations_G,
                    name="phonon bath")
system_G = oqupy.TimeDependentSystemWithField(
        h_sys_G)
mean_field_system_G = oqupy.MeanFieldSystem([system_G], field_eom_G)
tempo_params_G = oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))

# -----------------------------------------------------------------------------

def test_tempo_backend_G():
    tempo_G = oqupy.MeanFieldTempo(
        mean_field_system_G,
        [bath_G],
        tempo_params_G,
        [initial_state_G],
        initial_field_G,
        start_time=0.0)
    tempo_G.compute(end_time=t_end_G)
    dyn_G = tempo_G.get_dynamics()
    np.testing.assert_almost_equal(dyn_G.fields[-1], field_G, decimal=4)
    np.testing.assert_almost_equal(
            dyn_G.system_dynamics[0].states[-1], rho_G, decimal=4)

def test_tensor_network_pt_tempo_backend_A():
    pt = oqupy.pt_tempo_compute(
        bath_G,
        start_time=0.0,
        end_time=t_end_G,
        parameters=tempo_params_G)

    dyn = oqupy.compute_dynamics_with_field(
        mean_field_system=mean_field_system_G,
        initial_field=initial_field_G,
        process_tensor_list=[pt],
        initial_state_list=[initial_state_G])
    np.testing.assert_almost_equal(dyn.fields[-1], field_G, decimal=4)
    np.testing.assert_almost_equal(
            dyn.system_dynamics[0].states[-1], rho_G, decimal=4)

# -----------------------------------------------------------------------------

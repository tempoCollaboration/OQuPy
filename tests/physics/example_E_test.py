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
# -- Test E: Superohmic independent spin boson model --------------------------

# Initial state:
initial_state_E = oqupy.operators.spin_dm("y+")

# System operator
h_sys_E = 0.0 * oqupy.operators.sigma("x")

# Ohmic spectral density with exponential cutoff
coupling_operator_E = 0.5 * oqupy.operators.sigma("z")
alpha_E = 0.3
cutoff_E = 5.0
temperature_E = 0.0

# end time
t_start_E = 0.0
t_end_E = 10.0

# exact result:
def exact_result(t):
    x = (t*cutoff_E)**2
    phi = 2 * alpha_E * (1 + (x-1)/(x+1)**2)
    y_plus = np.exp(-phi)
    dm = (1 - y_plus) * oqupy.operators.spin_dm("mixed") \
         + y_plus * oqupy.operators.spin_dm("y+")
    return dm

rho_E = exact_result(t_end_E)

correlations_E = oqupy.PowerLawSD(
    alpha=alpha_E,
    zeta=3.0,
    cutoff=cutoff_E,
    cutoff_type="exponential",
    temperature=temperature_E,
    name="superohmic")
bath_E = oqupy.Bath(
    coupling_operator_E,
    correlations_E,
    name="superohmic phonon bath")
system_E = oqupy.System(h_sys_E)

# -----------------------------------------------------------------------------

def test_tensor_network_tempo_backend_E():
    tempo_params_E = oqupy.TempoParameters(
        dt=0.4,
        dkmax=2,
        epsrel=1.0e-5,
        add_correlation_time=np.infty)
    tempo_E = oqupy.Tempo(
        system_E,
        bath_E,
        tempo_params_E,
        initial_state_E,
        start_time=t_start_E)
    tempo_E.compute(end_time=t_end_E)
    dyn_E = tempo_E.get_dynamics()
    np.testing.assert_almost_equal(dyn_E.states[-1], rho_E, decimal=4)


def test_tensor_network_pt_tempo_backend_E():
    tempo_params_E = oqupy.TempoParameters(
        dt=0.4,
        dkmax=2,
        epsrel=1.0e-5,
        add_correlation_time=np.infty)
    pt = oqupy.pt_tempo_compute(
        bath_E,
        start_time=t_start_E,
        end_time=t_end_E,
        parameters=tempo_params_E)

    dyn = oqupy.compute_dynamics(
        system=system_E,
        process_tensor=pt,
        initial_state=initial_state_E)
    np.testing.assert_almost_equal(dyn.states[-1], rho_E, decimal=4)

# -----------------------------------------------------------------------------

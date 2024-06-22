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
Tests for GibbsTEMPO.
"""

import pytest
import numpy as np

import oqupy

# -----------------------------------------------------------------------------

h_sys = np.array([[0.5, 0.0], [0, -0.5]])
cutoff = 5.0
temperature = 2.1

indep_sbm = lambda a: 1 / 2 \
    * (np.eye(2) \
       - 2 * np.tanh(1 / (2 * temperature) - a * cutoff / temperature) * h_sys)

system = oqupy.System(h_sys)

gibbs_params = oqupy.GibbsParameters(
    n_steps=100,
    epsrel=10**(-11))

V = np.array([[1.0 , 1.0],[1.0, -1.0]]) / np.sqrt(2)
def transf(matrix):
    return V@matrix@V.T.conj()

# -----------------------------------------------------------------------------

# -- Test A: Zero coupling --
system_A = oqupy.System(h_sys)
alpha_A = 0.0
coupling_operator_A = np.array([[1.0, 0.0], [0.0, 0.0]])
correlations_A = oqupy.PowerLawSD(
    alpha=alpha_A,
    zeta=1.0,
    cutoff=cutoff,
    cutoff_type="exponential",
    temperature=temperature,
    name="ohmic")
bath_A = oqupy.Bath(
    coupling_operator_A,
    correlations_A,
    name="phonon bath")


# -- Test B: Independent boson model --
system_B = oqupy.System(h_sys)
alpha_B = 0.3
coupling_operator_B = np.array([[1.0, 0.0], [0.0, 0.0]])
correlations_B = oqupy.PowerLawSD(
    alpha=alpha_B,
    zeta=1.0,
    cutoff=cutoff,
    cutoff_type="exponential",
    temperature=temperature,
    name="ohmic")
bath_B = oqupy.Bath(
    coupling_operator_B,
    correlations_B,
    name="phonon bath")

# -- Test C: Transformed independent boson model --
system_C = oqupy.System(transf(h_sys))
alpha_C = alpha_B
coupling_operator_C = transf(coupling_operator_B)
correlations_C = oqupy.PowerLawSD(
    alpha=alpha_C,
    zeta=1.0,
    cutoff=cutoff,
    cutoff_type="exponential",
    temperature=temperature,
    name="ohmic")
bath_C = oqupy.Bath(
    coupling_operator_C,
    correlations_C,
    name="phonon bath")

# -----------------------------------------------------------------------------

def test_gibbs_backend_A():
    tempo_A = oqupy.GibbsTempo(
        system_A,
        bath_A,
        gibbs_params)
    tempo_A.compute()
    gibbs_A = tempo_A.get_state()
    np.testing.assert_almost_equal(gibbs_A, indep_sbm(alpha_A), decimal=8)

def test_gibbs_backend_B():
    gibbs_B = oqupy.gibbs_tempo_compute(
        system_B,
        bath_B,
        gibbs_params)
    np.testing.assert_almost_equal(gibbs_B, indep_sbm(alpha_B), decimal=8)

@pytest.mark.skip(reason="See GitHub Issue#???")
def test_gibbs_backend_C():
    gibbs_C = oqupy.gibbs_tempo_compute(
        system_C,
        bath_C,
        gibbs_params)
    np.testing.assert_almost_equal(gibbs_C, transf(indep_sbm(alpha_C)), decimal=8)

# -----------------------------------------------------------------------------

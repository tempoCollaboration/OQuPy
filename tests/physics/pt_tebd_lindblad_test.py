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
Tests for PT-TEBD with Lindblad dissipators.
"""

import sys
sys.path.insert(0,'.')

import pytest
import numpy as np

import oqupy

# -----------------------------------------------------------------------------
# -- Test F: Test Lindblad dissipation for PT-TEBD  ---------------------------

# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.1
num_steps = 10

# -- bath --
alpha = 0.3
omega_cutoff = 3.0
temperature = 0.8
pt_epsrel = 1.0e-6

# -- chain --
N = 5
Omega = 1.0
eta = 0.3
Delta = 1.2
h = np.array(
    [[1.0, 0.0, 0.0],
     [2.0, 0.0, 0.0],
     [3.0, 0.0, 0.0],
     [4.0, 0.0, 0.0],
     [5.0, 0.0, 0.0]]) * np.pi / 10
J = np.array([[Delta, 1.0+eta, 1.0-eta]]*(N-1))
up_dm = oqupy.operators.spin_dm("z+")
down_dm = oqupy.operators.spin_dm("z-")
tebd_order = 2
tebd_epsrel = 1.0e-7


def test_pt_tebd_site_dissipation_H1():
    # -- initial state --
    initial_augmented_mps = oqupy.AugmentedMPS([up_dm, down_dm, down_dm])

    # -- add single site dissipation --
    system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2,2,2])
    # lowering operator on site 0:
    system_chain.add_site_dissipation(0,[[0,0],[1,0]])
    # identity cross raising operator on sites 1 and 2:
    system_chain.add_nn_dissipation(1,np.identity(2),[[0,1],[0,0]])

    # -- PT-TEBD parameters --
    pt_tebd_params = oqupy.PtTebdParameters(
        dt=dt,
        order=tebd_order,
        epsrel=tebd_epsrel)

    num_steps = int(1.0/pt_tebd_params.dt)

    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=[None]*3,
        parameters=pt_tebd_params,
        dynamics_sites=[0,1,2],
        chain_control=None)

    r = pt_tebd.compute(num_steps, progress_type="silent")

    np.testing.assert_almost_equal(
        r['dynamics'][0].states[-1],
        [[np.exp(-1),0],[0,1-np.exp(-1)]],
        decimal=4)
    np.testing.assert_almost_equal(
        r['dynamics'][1].states[-1],
        [[0,0],[0,1]],
        decimal=4)
    np.testing.assert_almost_equal(
        r['dynamics'][2].states[-1],
        [[1-np.exp(-1),0],[0,np.exp(-1)]],
        decimal=4)

# -----------------------------------------------------------------------------

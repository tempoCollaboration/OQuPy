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
import sys
sys.path.insert(0,'.')

import pytest
import numpy as np

import oqupy

# -----------------------------------------------------------------------------
# -- Test F: XYZ Spin chain without and with bath  ----------------------------

# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.1
num_steps = 10

# -- bath --
alpha = 0.3
omega_cutoff = 3.0
temperature = 0.8
pt_dkmax = 10
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
initial_state_site = oqupy.operators.spin_dm("z-")
tebd_order = 2
tebd_epsrel = 1.0e-7


# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=3,
    cutoff=omega_cutoff,
    cutoff_type='exponential',
    temperature=temperature)
bath = oqupy.Bath(0.5 * oqupy.operators.sigma("z"), correlations)
tempo_parameters = oqupy.TempoParameters(
    dt=dt,
    dkmax=pt_dkmax,
    epsrel=pt_epsrel,
    add_correlation_time=5.0)
pt = oqupy.pt_tempo_compute(
    bath=bath,
    start_time=0.0,
    end_time=num_steps * dt,
    parameters=tempo_parameters,
    progress_type='bar')

# --- PT-TEBD preperation -----------------------------------------------------

# -- initial state --
initial_augmented_mps = oqupy.AugmentedMPS([initial_state_site] * N)

# -- chain hamiltonian --
system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2]*N)

for n in range(N):
    for i, xyz in enumerate(["x", "y", "z"]):
        system_chain.add_site_hamiltonian(
            site=n,
            hamiltonian=0.5*h[n, i]*oqupy.operators.sigma(xyz))
for n in range(N-1):
    for i, xyz in enumerate(["x", "y", "z"]):
        system_chain.add_nn_hamiltonian(
            site=n,
            hamiltonian_l=0.5*J[n, i]*oqupy.operators.sigma(xyz),
            hamiltonian_r=0.5*oqupy.operators.sigma(xyz))

pt_tebd_params = oqupy.PtTebdParameters(
    dt=dt,
    order=tebd_order,
    epsrel=tebd_epsrel)
dynamics_sites=list(range(N))

# -----------------------------------------------------------------------------

correct_F1_rhos = np.load("tests/data/example_F1_rhos.npy")
correct_F2_rhos = np.load("tests/data/example_F2_rhos.npy")

# -----------------------------------------------------------------------------

def test_pt_tebd_backend_F1():
    process_tensors = [None] * N
    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=process_tensors,
        parameters=pt_tebd_params,
        dynamics_sites=dynamics_sites,
        chain_control=None)

    r = pt_tebd.compute(num_steps, progress_type="silent")
    for site in dynamics_sites:
        pt_tebd_rho = r['dynamics'][site].states[-1]
        correct_rho = correct_F1_rhos[site]
        np.testing.assert_almost_equal(pt_tebd_rho, correct_rho, decimal=4)

# -----------------------------------------------------------------------------

def test_pt_tebd_backend_F2():
    process_tensors = [pt, None, None, pt, None]
    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=process_tensors,
        parameters=pt_tebd_params,
        dynamics_sites=dynamics_sites,
        chain_control=None)

    r = pt_tebd.compute(num_steps, progress_type="silent")
    for site in dynamics_sites:
        pt_tebd_rho = r['dynamics'][site].states[-1]
        correct_rho = correct_F2_rhos[site]
        np.testing.assert_almost_equal(pt_tebd_rho, correct_rho, decimal=4)

# -----------------------------------------------------------------------------

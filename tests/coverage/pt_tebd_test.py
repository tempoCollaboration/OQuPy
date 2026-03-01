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
Tests for the time_evovling_mpo.pt_tebd module.
"""

import pytest

import numpy as np
import oqupy



def test_get_augmented_mps_A():
    up_dm = oqupy.operators.spin_dm("z+")
    system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2,3])
    initial_augmented_mps = oqupy.AugmentedMPS([up_dm, np.diag([1,0,0])])
    pt_tebd_params = oqupy.PtTebdParameters(dt=0.2, order=2, epsrel=1.0e-4)

    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=[None, None],
        parameters=pt_tebd_params)

    augmented_mps = pt_tebd.get_augmented_mps()
    assert augmented_mps.gammas[0].shape == (1,4,1,1)
    assert augmented_mps.gammas[1].shape == (1,9,1,1)

    pt_tebd.compute(end_step=2, progress_type='silent')
    augmented_mps = pt_tebd.get_augmented_mps()
    assert augmented_mps.gammas[0].shape == (1,4,1,1)
    assert augmented_mps.gammas[1].shape == (1,9,1,1)


def test_get_augmented_mps_B():

    dt = 0.2
    epsrel = 1.0e-4
    num_steps = 4


    sx = 0.5 * oqupy.operators.sigma("x")
    sy = 0.5 * oqupy.operators.sigma("y")
    sz = 0.5 * oqupy.operators.sigma("z")
    up_dm = oqupy.operators.spin_dm("z+")
    down_dm = oqupy.operators.spin_dm("z-")
    mixed_dm = oqupy.operators.spin_dm("mixed")

    #System
    N = 3 # >=3
    epsilon = 1
    J_gamma = [1.3, 0.7, 1.2]
    system_chain = oqupy.SystemChain([2]*N)

    for n in range(N):
        system_chain.add_site_hamiltonian(site = n, hamiltonian= epsilon*sz)

    for n in range(N-1):
        for J, S in zip (J_gamma ,[sx,sy,sz]):
            system_chain.add_nn_hamiltonian(site = n, 
                                            hamiltonian_l = J*S,
                                            hamiltonian_r = S)

    #parameters
    pt_tebd_params = oqupy.PtTebdParameters(dt = dt,
                                        order = 2,
                                        epsrel = epsrel)

    dynamics_sites = list(range(0, N))
    initial_augmented_mps = oqupy.AugmentedMPS([up_dm]*N) 


    correlations = oqupy.PowerLawSD(alpha=0.05,
                                    zeta=1,
                                    cutoff=1.0,
                                    cutoff_type='exponential',
                                    temperature=0.0)
    bath = oqupy.Bath(0.5 * sy, correlations)
    pt_tempo_parameters = oqupy.TempoParameters(dt=dt,
                                                epsrel=1.0e-4,
                                                dkmax=10)

    pt = oqupy.pt_tempo_compute(bath=bath,
                                start_time=0.0,
                                end_time=num_steps * dt,
                                parameters=pt_tempo_parameters,
                                progress_type='bar')


    pt_tebd = oqupy.PtTebd(initial_augmented_mps = initial_augmented_mps,
                            system_chain = system_chain,
                            process_tensors = [None]+[pt]+[None]*(N-2),
                            parameters = pt_tebd_params,
                            dynamics_sites = dynamics_sites,
                            chain_control=None
                            )

    end_step = 2
    pt_tebd.compute(end_step = 2, progress_type = 'bar')

    new_augmented_mps = pt_tebd.get_augmented_mps()
    assert new_augmented_mps.gammas[1].shape[2] == pt.get_bond_dimensions()[end_step]

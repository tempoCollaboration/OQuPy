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
Performance tests for PT-TEBD computations.
"""
import sys
sys.path.insert(0,'.')
import os

import numpy as np
import time

import oqupy
import oqupy.operators as op
from tests.data.generate_pts import PT_DIR_PATH


# -- Test A -------------------------------------------------------------------

def pt_tebd_performance_A(process_tensor_name,
                          number_of_sites,
                          number_of_pts,
                          model,
                          tebd_epsrel):
    """
    Here we compute the dynamics of a XY or XYZ Heisenberg spin chain with
    environments coupling strongly to the first few sites of the chain.

    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
    number_of_sites:
        Total number of sites in the XY / XYZ spin chain.
    number_of_pts:
        Number of the spins that couple to an environment.
    model:
        Which model to use: 'XY' or 'XYZ'.
    tebd_epsrel:
        Relative SVD cutoff of the PT-TEBD algorithm.
"""
    N = number_of_sites
    K = number_of_pts
    tebd_order = 2


    # -- process tensor --
    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    pt = oqupy.import_process_tensor(pt_file_path)

    # -- chain hamiltonian --
    if model=="XY":
        h = np.array([[0.0, 0.0, 1.0]]*N)
        J = np.array([[0.96, 1.04, 0.0]]*(N-1))
    elif model=="XYZ":
        h = np.array([[0.0, 0.0, 1.0]]*N)
        J = np.array([[1.3, 0.7, 1.2]]*(N-1))
    else:
        raise ValueError(f"Model '{model}' not implemented!")

    system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2]*N)

    for n in range(N):
        for i, xyz in enumerate(["x", "y", "z"]):
            system_chain.add_site_hamiltonian(
                site=n,
                hamiltonian=0.5*h[n, i]*op.sigma(xyz))
    for n in range(N-1):
        for i, xyz in enumerate(["x", "y", "z"]):
            system_chain.add_nn_hamiltonian(
                site=n,
                hamiltonian_l=0.5*J[n, i]*op.sigma(xyz),
                hamiltonian_r=0.5*op.sigma(xyz))

    # -- initial state --
    initial_augmented_mps = oqupy.AugmentedMPS(
        [op.spin_dm("z+")] + [op.spin_dm("z-")] * (N-1))

    # -- pt-tebd computation --
    pt_tebd_params = oqupy.PtTebdParameters(
        dt=pt.dt,
        order=tebd_order,
        epsrel=tebd_epsrel)

    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=[pt]*K+[None]*(N-K),
        parameters=pt_tebd_params,
        dynamics_sites=list(range(N)))

    start_time = time.time()
    result = pt_tebd.compute(len(pt))
    result['walltime'] = time.time()-start_time
    result['N'] = N
    result['K'] = K
    result['model'] = model
    result['tebd_esprel'] = tebd_epsrel

    return result

# Parameter set intended to check convergence with PTs epsrel.
parameters_A1 = [
    ["spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel15",
     "spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel16",
     "spinBoson_alpha0.16_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel17"], # process_tensor_name
    [7],                             # number_of_sites
    [2],                             # number_of_pts
    ["XY"],                          # model
    [2**(-15), 2**(-16), 2**(-17)],  # tebd_epsrel
]

# Parameter set intended study the scaling of bond dimension, walltime etc
# for an increasing number of coupled environments 'number_of_pts'.
parameters_A2 = [
    ["spinBoson_alpha0.32_zeta1.0_T0.0_cutoff1.0exp_tcut4.0_dt04_steps06_epsrel16"], # process_tensor_name
    [7],                     # number_of_sites
    [0,1,2,3],               # number_of_pts
    ["XY"],                  # model
    [2**(-16)],              # tebd_epsrel
]

# -----------------------------------------------------------------------------

ALL_TESTS = [
    (pt_tebd_performance_A, [parameters_A1, parameters_A2]),
    # (pt_tebd_performance_B, [parameters_B1, parameters_B2]),
]

REQUIRED_PTS = list(set().union(*[params[0] for params in [parameters_A1, parameters_A2]]))

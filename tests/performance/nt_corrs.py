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
Performance tests for multi-time correlation computations.
"""
import sys
sys.path.insert(0,'.')
import os

import numpy as np
import time

import oqupy


PT_DIR_PATH = "./tests/data/process_tensors/"

# -- Test A -------------------------------------------------------------------

def nt_corr_performance_A(process_tensor_name,
                          number_of_times,
                          pt_epsrel):
    """
    Here we compute a four-time correlation of a three-level system with the
    excited states coupled to a bosonic bath.

    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
    number_of_times:
        Total number of time arguments (exlcuding the final time) at which to
        evaluate the correlation function.
    pt_epsrel:
        Relative SVD cutoff of the PT-TEMPO algorithm.
"""
    N = number_of_times

    # -- process tensor --
    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    pt = oqupy.import_process_tensor(pt_file_path)

    # -- system hamiltonian --
    P_1 = np.array([[0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.]], dtype=complex)

    P_2 = np.array([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 1.]], dtype=complex)

    sigma_min = np.array([[0., 0., 0.],
                          [0., 0., 1.],
                          [0., 0., 0.]], dtype=complex)

    sigma_plus = np.array([[0., 0., 0.],
                           [0., 0., 0.],
                           [0., 1., 0.]], dtype=complex)

    system = oqupy.System((5.)*(P_1 + P_2)
                          + 2. * (sigma_plus + sigma_min))

    # -- initial state --
    initial_state = np.array([[1., 0., 0.],
           [0., 0., 0.],[0., 0., 0.]], dtype=complex)

    # -- correlation computation --
    dip_v = np.array([[0., 0., 1.],
           [0., 0., 0.],[1., 0., 0.]], dtype=complex)
    operators = [dip_v, dip_v, dip_v, dip_v]
    ops_order = ["left", "right", "right", "left"]

    dt = pt.dt
    start_t = 0.

    times_1 = slice(0,N)
    times_2 = slice(N-1,N)
    times_3 = slice(N-1, N)
    times_4 = slice(N-1, len(pt))

    ops_times = [times_1, times_2, times_3, times_4]

    keys = ['corrs', 'walltime', 'N', 'pt_epsrel']
    result = dict.fromkeys(keys, None)

    start_time = time.time()
    cor = oqupy.compute_correlations_nt(
        system = system,
        process_tensor=pt,
        operators = operators,
        ops_times=ops_times,
        ops_order=ops_order,
        dt = dt,
        initial_state = initial_state,
        start_time = start_t,
        progress_type = "bar")

    result['corrs'] = cor[1][:,0,0,:]
    result['walltime'] = time.time()-start_time
    result['N'] = N
    result['pt_esprel'] = pt_epsrel

    return result

# Parameter set intended to check convergence with PTs epsrel.
parameters_A1 = [
    ["3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel4",
     "3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel6",
     "3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel8"], # process_tensor_name
    [1],                             # number_of_times
    [10**(-4), 10**(-6), 10**(-8)],  # pt_epsrel
]

# Parameter set intended study the scaling of bond dimension, walltime etc
# for an increasing number of time arguments 'number_of_times'.
parameters_A2 = [
    ["3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel6"], # process_tensor_name
    [1,2,3,4,5,6,7,8,9,10],  # number_of_times
    [10**(-6)],              # pt_epsrel
]

# -----------------------------------------------------------------------------

ALL_TESTS = [
    (nt_corr_performance_A, [parameters_A1, parameters_A2]),
]

REQUIRED_PTS = list(set().union(*[params[0] for params in [parameters_A1, parameters_A2]]))

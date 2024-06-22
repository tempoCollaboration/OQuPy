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
Performance tests for bath dynamics computations.
"""
import sys
sys.path.insert(0,'.')
import os

import numpy as np
import time

import oqupy
import oqupy.operators as op
from tests.data.generate_pts import PT_DIR_PATH
from tests.data.generate_pts import parameters_from_name, bath_from_parameters

# -- Test A -------------------------------------------------------------------

def bath_dynamics_performance_A(process_tensor_name,
                                epsilon,
                                Omega):
    """
    ??Here we compute the dynamics of a XY or XYZ Heisenberg spin chain with
    environments coupling strongly to the first few sites of the chain.

    ??Parameters
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

    # -- process tensor --
    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    pt = oqupy.import_process_tensor(pt_file_path)

    # -- system and bath --
    system = oqupy.System(0.5*Omega*op.sigma('x') + 0.5*epsilon*op.sigma('z'))
    p = parameters_from_name(process_tensor_name)
    bath = bath_from_parameters(p)
    initial_state = op.spin_dm('down')
    bath_corr = oqupy.bath_dynamics.TwoTimeBathCorrelations(
        system,
        bath,
        pt,
        initial_state)

    end_time = 2**(p['stepsExp']-p['dtExp'])

    # -- ?? --
    w_start = 1.0
    w_stop = 3.0
    w_step = 0.05
    # w2index = int((2.0-w_start)/w_inc)

    ws = np.arange(w_start, w_stop, w_step)
    energies = np.full((len(ws),len(pt)+1),np.nan)
    energiesEnvelope = np.full((len(ws),len(pt)+1),np.nan)

    start_time = time.time()
    bath_corr.generate_system_correlations(final_time=end_time)
    generate_correlations_time = time.time() - start_time

    for i, w in enumerate(ws):
        delta = 0.1
        ts, occ = bath_corr.occupation(w, delta, change_only = True)
        energy = w * occ
        steps = int(2*np.pi/(w*pt.dt))
        window = np.ones(steps)/steps
        energies[i,:] = energy
        energiesEnvelope[i,:] = np.convolve(energy,window,mode='same')
        energiesEnvelope[i,:steps//2] = np.nan
        energiesEnvelope[i,-steps//2:] = np.nan


    result = dict()
    result['process_tensor_name'] = process_tensor_name
    result['epsilon'] = epsilon
    result['Omega'] = Omega
    result['walltime'] = time.time()-start_time
    result['generate_correlations_time'] = generate_correlations_time
    result['w_start_stop_step'] = (w_start, w_stop, w_step)
    result['ws'] = ws
    result['ts'] = ts
    result['energies'] = energies
    result['energiesEnvelope'] = energiesEnvelope

    return result

# Parameter set intended to check convergence with PTs epsrel.
parameters_A1 = [
    ["spinBoson_alpha0.05_zeta1.0_T1.0_cutoff10.0exp_tcut1.0_dt03_steps08_epsrel15"], # process_tensor_name
    [2.0],                # epsilon
    [1.0],                # Omega
]

# -----------------------------------------------------------------------------

ALL_TESTS = [
    (bath_dynamics_performance_A, [parameters_A1]),
]

REQUIRED_PTS = list(set().union(*[params[0] for params in [parameters_A1]]))

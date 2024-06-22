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
Performance tests for multiple environment computations.
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

def multi_env_performance_A(process_tensor_names, initial_state_name):
    """
    Here we compute the dynamics of a two level system coupled to one or more environments.

    Parameters
    ----------
    process_tensor_names:
        Names of the process tensors that represents the environments (following
        the convention defined in /tests/data/generate_pts.py)
    initial_state_name:
        Name of the initial system state compatible with
        `oqupy.operators.spin_dm(initial_state_name)`
"""

    # -- process tensors --
    pts = []
    for process_tensor_name in process_tensor_names:
        pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
        pts.append(oqupy.import_process_tensor(pt_file_path))

    # -- system hamiltonian --
    epsilon = 2.0
    Omega = 1.0
    system = oqupy.System(0.5*epsilon*op.sigma('z') + 0.5*Omega*op.sigma('x'))

    # -- initial state --
    initial_state = op.spin_dm(initial_state_name)

    # -- compute dynamics --
    start_time = time.time()

    dynamics = oqupy.compute_dynamics(
        system,
        process_tensor=pts,
        initial_state=initial_state)

    result = dict()
    result['walltime'] = time.time()-start_time
    result['process_tensor_names'] = process_tensor_names
    result['initial_state_name'] = initial_state_name
    result['dynamics'] = dynamics

    return result

# Parameter set to compare equilibration with 1) cold bath, 2) hot bath, 3) cold and hot bath.
parameters_A1 = [
    [["spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_z"],
     ["spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_x"],
     ["spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_z",
      "spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_x"]
    ],
    ['z+', 'z-'],                    # number_of_sites
]

parameters_A2 = [
    [["spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_z"],
     ["spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_x"],
     ["spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_z",
      "spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_x"]
    ],
    ['z+', 'z-'],                    # number_of_sites
]

# -----------------------------------------------------------------------------

ALL_TESTS = [
    (multi_env_performance_A, [parameters_A2]),
]

REQUIRED_PTS = [
    # "spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_z",
    # "spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt04_steps08_epsrel17_x",
    "spinBoson_alpha0.16_zeta1.0_T0.8_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_z",
    "spinBoson_alpha0.16_zeta1.0_T1.6_cutoff1.0exp_tcut4.0_dt03_steps09_epsrel15_x"
]

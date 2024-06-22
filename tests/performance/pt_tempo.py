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
from tests.data.generate_pts import generate_spin_boson_pt
from tests.data.generate_pts import parameters_from_name
from tests.data.generate_pts import bath_from_parameters

# -- Test A -------------------------------------------------------------------

def pt_tempo_A(process_tensor_name):
    """
    Here we compute the dynamics of a two level system coupled to an 
    environment.

    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
"""

    params = parameters_from_name(process_tensor_name)

    # -- compute process tensor --
    t0 = time.time()
    generate_spin_boson_pt(process_tensor_name)
    t1 = time.time()

    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    pt = oqupy.import_process_tensor(pt_file_path)

    # -- system hamiltonian --
    Omega = 1.0
    system = oqupy.System(0.5*Omega*op.sigma('x'))

    # -- initial state --
    initial_state = op.spin_dm('z+')

    # -- compute dynamics --

    t2 = time.time()
    dynamics = oqupy.compute_dynamics(
        system,
        process_tensor=pt,
        initial_state=initial_state)
    t3 = time.time()

    result = dict()
    result['pt_tempo_walltime'] = t1-t0
    result['dynamics_walltime'] = t3-t2
    result['process_tensor_name'] = process_tensor_name
    result['params'] = params
    result['dynamics'] = dynamics

    return result

parameters_A1 = [
    ["spinBoson_alpha0.10_zeta1.0_T0.0_cutoff5.0exp_tcut4.0_dt04_steps08_epsrel14",
     "spinBoson_alpha0.30_zeta1.0_T0.0_cutoff5.0exp_tcut4.0_dt04_steps08_epsrel14",
     "spinBoson_alpha0.70_zeta1.0_T0.0_cutoff5.0exp_tcut4.0_dt04_steps08_epsrel14",
     "spinBoson_alpha1.10_zeta1.0_T0.0_cutoff5.0exp_tcut4.0_dt04_steps08_epsrel14",
    ],
]

def pt_tempo_B(process_tensor_name, spin_dim):
    """
    Here we compute the dynamics of a spin with dimension `spin_dim`
    coupled to an environment using / not using the degenaracy checking.

    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
    spin_dim:
        Hilbert space dimension of the spin.
"""

    p = parameters_from_name(process_tensor_name)

    cutoff_map = {'gss': 'gaussian', 'exp': 'exponential', 'hrd':'hard'}
    try:
        cutoff_type = cutoff_map[p['cutoffType']]
    except KeyError:
        raise ValueError(f"Cutoff Type {p['cutoffType']} not known.")

    correlations = oqupy.PowerLawSD(alpha=p['alpha'],
                                    zeta=p['zeta'],
                                    cutoff=p['cutoff'],
                                    cutoff_type=cutoff_type,
                                    temperature=p['T'])
    
    coupling_op = np.diag(-(np.arange(spin_dim)-(spin_dim-1)/2))
    bath = oqupy.Bath(coupling_op, correlations)
    
    dt = 2**(-p['dtExp'])
    steps = 2**p['stepsExp']
    epsrel = 2**(-p['epsrelExp'])
    end_time = 2**(p['stepsExp']-p['dtExp'])

    pt_tempo_parameters = oqupy.TempoParameters(dt=dt, tcut=p['tcut'], epsrel=epsrel)

    t0 = time.time()
    ptA = oqupy.pt_tempo_compute(
        bath=bath,
        start_time=0.0,
        unique=False,
        end_time=end_time,
        parameters=pt_tempo_parameters,
        overwrite=True,
    )
    t1 = time.time()

    t2 = time.time()
    ptB = oqupy.pt_tempo_compute(
        bath=bath,
        start_time=0.0,
        unique=True,
        end_time=end_time,
        parameters=pt_tempo_parameters,
        overwrite=True,
    )
    t3 = time.time()

    result = dict()
    result['ptA_walltime'] = t1-t0
    result['ptB_walltime'] = t3-t2
    result['bond_dim_A'] = ptA.get_bond_dimensions()
    result['bond_dim_B'] = ptB.get_bond_dimensions()
    result['params'] = p

    return result

parameters_B1 = [
    ["spinBoson_alpha0.10_zeta1.0_T1.0_cutoff1.0exp_tcut2.0_dt04_steps05_epsrel14"],
    [2,3,4,5,6,7]
]

# -----------------------------------------------------------------------------

ALL_TESTS = [
    (pt_tempo_A, [parameters_A1]),
    (pt_tempo_B, [parameters_B1]),
]

REQUIRED_PTS = []

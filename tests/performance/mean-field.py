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
# """
# Performance tests for PT-TEBD computations.
# """
import sys
sys.path.insert(0,'.')
import os

import dill
import itertools
import numpy as np
import time


import oqupy
import oqupy.operators as op


PT_DIR_PATH = "./tests/data/process_tensors/"

# -- Test A -------------------------------------------------------------------
# """
# Computational cost with number of systems.
# """

# dt = 3.2e-3 = 3.2 fs (converged ~ 1 fs)
# steps 2**6 -> 0.2 ps, probably file
# epsrel: 10**-6 = 2**P? 

parameters_A1 = [
    ["spinBoson_alpha0.25_zeta1.0_T39.3_cutoff1.0expon_tcut227.9_dt04_steps06_epsrel15"],
    [i for i in range(1, 11)],      # number of systems (up to 10 systems)
]


def mean_field_performance_A(process_tensor_name, 
                             number_of_systems):
    """
    ToDo
    """

    # define system
    # hardcode in system parameters

    mean_field_system = oqupy.MeanFieldSystem(system_list=[],
                                              field_eom=field_eom)

    start_time = time.time()
    mean_field_dynamics = oqupy.compute_dynamics_with_field(
        process_tensor_list=number_of_systems * [process_tensor],
        mean_field_system=mean_field_system,
        initial_state_list=[initial_state],
        initial_field=initial_field,
        start_time=0.0)
    result['walltime'] = time.time()-start_time
    result['N'] = number_of_sites
    t, fields = mean_field_dynamics.field_expectations()
    result['t'] = t
    result['fields'] = fields
    # e.g. system0_States np.allclose() system-1 states
    #system0_dynamics = mean_field_dynamics.system_dynamics[0] 
    #system0_states = system0_dynamics._states

    return result


# -----------------------------------------------------------------------------

ALL_TESTS = [
    (mean_field_performance_A, [parameters_A1]),
]

# -----------------------------------------------------------------------------

def run_all():
    results_list_list = []
    for performance_function, parameters_list in ALL_TESTS:
        results_list = []
        for parameters in parameters_list:
            param_comb = list(itertools.product(*parameters))
            results = [performance_function(*params) for params in param_comb]
            results_list.append(results)
        results_list_list.append(results_list)
    return results_list_list

# -----------------------------------------------------------------------------

all_results = run_all()
with open('./tests/data/temp/mean-field_results.pkl', 'wb') as f:
    dill.dump(all_results, f)


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
Performance tests for PT-TEBD computations.
"""

import itertools
# import numpy as np
# import oqupy


# -- Test A -------------------------------------------------------------------
"""
ToDo: Here should be some description of the performance tests scenario A.
"""

parameters_A1 = [
    ["boson_alpha0.08_zeta3.0_T0.0_gauss_dt0.04",
     "boson_alpha0.16_zeta3.0_T0.0_gauss_dt0.04",
     "boson_alpha0.32_zeta3.0_T0.0_gauss_dt0.04"], # process_tensor_name
    [8],                                           # number_of_sites
    ["XY", "XYZ"],                                 # model
    [1.0e-5, 1.0e-6, 1.0e-7],                      # tebd_epsrel
]

parameters_A2 = [
    ["boson_alpha0.16_zeta3.0_T0.0_gauss_dt0.04"], # process_tensor_name
    [4,8,12,16],                                   # number_of_sites
    [1.0e-6],                                      # tebd_epsrel
]

def pt_tebd_performance_A(process_tensor_name,
                          number_of_sites,
                          model,
                          tebd_epsrel):
    """
    ToDo: Implement computation A
    ToDo: This function should return the most relevant information concerning
          performance of a computation.
    """
    return NotImplemented


# -- Test B -------------------------------------------------------------------
"""
ToDo: Here should be some description of the performance tests scenario B.
"""

parameters_B1 = [
    ["boson_alpha0.16_zeta3.0_T0.8_gauss_dt0.04"], # left_process_tensor_name
    ["boson_alpha0.16_zeta3.0_T1.6_gauss_dt0.04"], # right_process_tensor_name
    [4,8,12,16],                                   # number_of_sites
    [1.0e-6],                                      # tebd_epsrel
]

def pt_tebd_performance_B(left_process_tensor_name,
                          right_process_tensor_name,
                          number_of_sites,
                          tebd_epsrel):
    """
    ToDo: implement computation B
    ToDo: This function should return the most relevant information concerning
          performance of a computation.
    """
    return NotImplemented


# -----------------------------------------------------------------------------

ALL_TESTS = [
    (pt_tebd_performance_A, [parameters_A1,parameters_A2]),
    (pt_tebd_performance_B, [parameters_B1]),
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


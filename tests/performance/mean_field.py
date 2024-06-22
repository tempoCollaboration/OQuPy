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
Performance tests for mean field TEMPO computations.
"""
import sys
sys.path.insert(0,'.')
import os

import numpy as np
import time

import oqupy


PT_DIR_PATH = "./tests/data/process_tensors/"

# -- Test A -------------------------------------------------------------------

def mean_field_performance_A(process_tensor_name, 
                             number_of_systems_list):
    """
    Checks runtime scaling with number of systems.
    Parameters
    ----------
    process_tensor_name:
        Name of the process tensor that represents the environment (following
        the convention defined in /tests/data/generate_pts.py)
    number_of_systems_list:
        List of number of distinct TimeDependentSystemWithField objects in each calculation.
        In addition, a calculation using a TimeDependentSystem is performed ('0' number_of_systems)
    
    Returned result contains times and field variables for each calculation, and the calculation's
    walltime.
    """

    # system parameters

    sigma_z = oqupy.operators.sigma("z")
    sigma_plus = oqupy.operators.sigma("+")
    sigma_minus = oqupy.operators.sigma("-")

    omega_0 = 0.0
    omega_c = -30.4
    Omega = 303.9

    kappa = 15.2
    Gamma_down = 15.2
    Gamma_up = 0.8 * Gamma_down

    gammas = [ lambda t: Gamma_down, lambda t: Gamma_up]
    lindblad_operators = [ lambda t: sigma_minus, lambda t: sigma_plus]

    def H_MF(t, a):
        return 0.5 * omega_0 * sigma_z +\
            0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)

    
    initial_field = np.sqrt(0.05)               # Note n_0 = <a^dagger a>(0) = 0.05
    initial_state = np.array([[0,0],[0,1]])     # spin down


    # load process tensor
    pt_file_path = os.path.join(PT_DIR_PATH, f"{process_tensor_name}.hdf5")
    process_tensor = oqupy.import_process_tensor(pt_file_path)

    result = {'times':[],
              'fields':[],
              'number_of_systems':[],
              'walltimes':[],
              'process_tensor_name':process_tensor_name,
              }

    def compute_n_systems(n):
        """Return runtime for the dynamics of mean field systems with n systems.
           (helper function) 
        """

        fractions = [1/n for i in range(n)]

        def field_eom(t, states, field):
            sx_exp_list = [np.matmul(sigma_minus, state).trace() for state in states]
            sx_exp_weighted_sum = sum([fraction*sx_exp for fraction, sx_exp in zip(fractions, sx_exp_list)])
            return -(1j*omega_c+kappa)*field - 0.5j*Omega*sx_exp_weighted_sum
        
        system = oqupy.TimeDependentSystemWithField(
            H_MF,
            gammas=gammas,
            lindblad_operators=lindblad_operators)
        system_list = [system for i in range(n)]
        mean_field_system = oqupy.MeanFieldSystem(system_list, field_eom=field_eom)

        start_time = time.perf_counter()
        dynamics = oqupy.compute_dynamics_with_field(
                                        mean_field_system, 
                                        initial_field=initial_field, 
                                        initial_state_list=[initial_state for i in range(n)], 
                                        start_time=0.0,
                                        process_tensor_list = [process_tensor for i in range(n)]
                                        )
        end_time = time.perf_counter()
        t, fields = dynamics.field_expectations()
        result['times'].append(t)
        result['fields'].append(fields)
        result['number_of_systems'].append(n)
        result['walltimes'].append(end_time - start_time)

    num_calculations = len(number_of_systems_list) + 1
    for i, n in enumerate(number_of_systems_list):
        print(f'#### inner-test calculation {i+1} of {num_calculations} (n={n} MeanFieldSystems)')
        compute_n_systems(n)
        if i > 0:
           assert np.allclose(result['fields'][i], result['fields'][0], atol=1e-4)

    # Now test with NO mean-field system
    target_t = result['times'][0]
    target_fields = result['fields'][0]
    target_field_dynamics = zip(target_t, target_fields)
    def H_MF_FAKE(t):
        a = next((field for time, field in target_field_dynamics if time >= t), target_fields[-1])
        return 0.5 * omega_0 * sigma_z +\
            0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)
    system = oqupy.TimeDependentSystem(H_MF_FAKE,
                                       gammas=gammas,
                                       lindblad_operators=lindblad_operators)

    print(f'#### inner-test calculation {num_calculations} of {num_calculations} (NO MeanFieldSystems)')
    start_time = time.perf_counter()
    dynamics = oqupy.compute_dynamics(
            system=system,
            process_tensor=process_tensor,
            start_time=0.0,
            initial_state=initial_state
            )
    end_time = time.perf_counter()
    result['times'].insert(0, dynamics.times)
    result['fields'].insert(0, target_fields)
    result['number_of_systems'].insert(0, 0)
    result['walltimes'].insert(0, end_time - start_time)

    return result

# -----------------------------------------------------------------------------
parameters_A1 = [
    #["spinBoson_alpha0.25_zeta1.0_T39.3_cutoff1.0exp_tcut227.9_dt10_steps06_epsrel15"], # easy
    ["spinBoson_alpha0.25_zeta1.0_T39.3_cutoff227.9exp_tcut0.1_dt10_steps07_epsrel26"], # realistic
    [[i for i in range(1, 11)]], # list of number of systems (up to 10 systems)
]

ALL_TESTS = [
    (mean_field_performance_A, [parameters_A1]),
]

# -----------------------------------------------------------------------------

REQUIRED_PTS = list(set().union(*[params[0] for params in [parameters_A1]]))


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

parameters_A1 = [
    ["spinBoson_alpha0.25_zeta1.0_T39.3_cutoff1.0expon_tcut227.9_dt10_steps06_epsrel15"],
    [[i for i in range(1, 11)]],      # list of number of systems (up to 10 systems)
]

parameters_B1 = [
    ["spinBoson_alpha0.25_zeta1.0_T39.3_cutoff1.0expon_tcut227.9_dt10_steps06_epsrel15"],
    [10]      # number of systems
]



def mean_field_performance_A(process_tensor_name, 
                             number_of_systems_list):
    """
    Checks runtime scaling with number of systems.
    Returns
    -------
        number_of_systems_list: list
            List with number of systems
        runtimes: list
            List with runtimes corresponding to different numbers of systems
    

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
        oqupy.compute_dynamics_with_field(
                                        mean_field_system, 
                                        initial_field=initial_field, 
                                        initial_state_list=[initial_state for i in range(n)], 
                                        start_time=0.0,
                                        process_tensor_list = [process_tensor for i in range(n)]
                                        )
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        return time_taken

    runtimes = [compute_n_systems(n) for n in number_of_systems_list]

    return number_of_systems_list, runtimes



def mean_field_performance_B(process_tensor_name, 
                             number_of_systems):
    """
    This test checks that all final subsystem states within a mean field system are the same.
    It returns the field expectations and associated times of the mean field system.

    Returns
    -------
        times: ndarray
            The points in time :math:`t`.
        field_expectations: ndarray
            Values :math:`$\langle`$ a(t) \rangle`.
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

    fractions = [1/number_of_systems for i in range(number_of_systems)]

    def field_eom(t, states, field):
        sx_exp_list = [np.matmul(sigma_minus, state).trace() for state in states]
        sx_exp_weighted_sum = sum([fraction*sx_exp for fraction, sx_exp in zip(fractions, sx_exp_list)])
        return -(1j*omega_c+kappa)*field - 0.5j*Omega*sx_exp_weighted_sum
    
    system = oqupy.TimeDependentSystemWithField(
        H_MF,
        gammas=gammas,
        lindblad_operators=lindblad_operators)
    system_list = [system for i in range(number_of_systems)]
    mean_field_system = oqupy.MeanFieldSystem(system_list, field_eom=field_eom)

    mean_field_dynamics = oqupy.compute_dynamics_with_field(
                                    mean_field_system, 
                                    initial_field=initial_field, 
                                    initial_state_list=[initial_state for i in range(number_of_systems)], 
                                    start_time=0.0,
                                    process_tensor_list = [process_tensor for i in range(number_of_systems)]
                                    )

    times, field_expectations = mean_field_dynamics.field_expectations()
    
    # check that all subsystem states are equal
    for i in range(number_of_systems - 1):
        assert np.allclose(mean_field_dynamics.system_dynamics[i].states, 
                        mean_field_dynamics.system_dynamics[i+1].states)
    
    # also check explicitly that first subsystem state is the same as last subsystem state
    # (extra check since np.allclose does not guarantee transitivity)
    assert np.allclose(mean_field_dynamics.system_dynamics[0].states, 
                        mean_field_dynamics.system_dynamics[-1].states)

  
    return times, field_expectations
# -----------------------------------------------------------------------------

ALL_TESTS = [
    (mean_field_performance_A, [parameters_A1]),
    (mean_field_performance_B, [parameters_B1])
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


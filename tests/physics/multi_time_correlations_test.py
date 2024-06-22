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
Tests multi-time system correlations.
"""
import sys
sys.path.insert(0,'.')

import pytest
import numpy as np

import oqupy


# -----------------------------------------------------------------------------
# -- Test: Multi-time correlations for independent boson model  ---------------

# --- Parameters --------------------------------------------------------------

sigma_x = oqupy.operators.sigma("x")
sigma_z = oqupy.operators.sigma("z")
down_density_matrix = oqupy.operators.spin_dm("z-")
up_density_matrix = oqupy.operators.spin_dm("z+")

eps = 5.
omega_cutoff = 3.
alpha = 0.1
temperature = 0.1309*10.
dt = 0.05
dkmax = 500
epsrel = 10**(-6)

start_time = 0.
end_time = dt*50

initial_state = down_density_matrix

#--------Define environment + compute process tensor---------------------------
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(up_density_matrix, correlations)

tempo_parameters = oqupy.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=start_time,
                                        end_time=end_time,
                                        parameters=tempo_parameters)

#---------------Define system Hamiltonian---------------
reorg = 2.0*alpha*omega_cutoff
system = oqupy.System(0.5 * (eps + reorg) * sigma_z)

#--------------------Analytical result---------------------------------------
g_t=[]
t_corr0=np.arange(0.,end_time+dt,dt)
for i in range(len(t_corr0)):
    g=correlations.correlation_2d_integral(delta=t_corr0[i],
                                           time_1=0.,shape="upper-triangle")
    g_t.append(g)

j_t=[]
for i in range (len(t_corr0)):
    j_ind = np.exp(-g_t[i]-1j*((reorg + eps)*t_corr0[i]))
    j_t.append(j_ind)
jt = np.array(j_t)


def test_nt_correlations_A():
    jt = np.array(j_t)
    j_t_num = oqupy.compute_correlations_nt(
        system = system,
        process_tensor = process_tensor,
        operators = [sigma_x, sigma_x],
        ops_times = [0. , (0.,end_time)],
        ops_order = ["left", "left"],
        dt = dt,
        initial_state = initial_state,
        start_time = start_time)
    jt_num = j_t_num[1][0]
    assert np.allclose(jt, jt_num, rtol = 1**(-6))



def test_compute_correlations_nt_D():
    ops_times = [0., 0., 0., 0.,]
    time_order = ["left", "left", "left", "left"]
    start_time = 0.
    operators = [sigma_x, sigma_x, sigma_x, sigma_x]

    system = oqupy.System(0.5 * eps * sigma_x)

    correlations = oqupy.PowerLawSD(
        alpha=alpha,
        zeta=1.0,
        cutoff=omega_cutoff,
        cutoff_type='exponential',
        temperature=temperature)
    bath = oqupy.Bath(0.5 * sigma_z, correlations)

    tempo_parameters = oqupy.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

    process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                            start_time=0.,
                                            end_time= dt * 10,
                                            parameters=tempo_parameters)

    cor = oqupy.compute_correlations_nt(
        system = system,
        process_tensor=process_tensor,
        operators = operators,
        ops_times=ops_times,
        ops_order=time_order,
        dt = dt,
        initial_state = initial_state,
        start_time = start_time,
        progress_type = "bar")

    assert np.isclose(cor[1][0][0][0][0], 1 + 0.j, rtol = 1**(-6))

#-------------Compare against compute_correlations---------------------------

def test_compute_correlations_nt_C():
    dt=0.1
    ops_times = [0., 10 * dt]
    time_order = ["left", "left"]
    start_time = 0.
    operators = [sigma_x, sigma_x]

    system = oqupy.System(0.5 * 5. * sigma_x)

    correlations = oqupy.PowerLawSD(alpha=0.1,
                                    zeta=1.,
                                    cutoff=3.,
                                    cutoff_type='exponential',
                                    temperature=1.)
    bath = oqupy.Bath(0.5 * sigma_z, correlations)

    tempo_parameters = oqupy.TempoParameters(dt=dt, dkmax=100, epsrel=10**(-4))

    process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                            start_time=0.,
                                            end_time= dt * 10,
                                            parameters=tempo_parameters)

    cor = oqupy.compute_correlations_nt(
        system = system,
        process_tensor=process_tensor,
        operators = operators,
        ops_times=ops_times,
        ops_order=time_order,
        dt = dt,
        initial_state = initial_state,
        start_time = start_time,
        progress_type = "bar")


    j_t = oqupy.compute_correlations(
        system = system,
        process_tensor = process_tensor,
        operator_a = sigma_x,
        operator_b = sigma_x,
        times_a = 0.0,
        times_b = dt*10,
        time_order = "ordered",
        dt = dt,
        initial_state = initial_state,
        start_time = 0.)
    assert np.isclose(cor[1][0], j_t[1][0])




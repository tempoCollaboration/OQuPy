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
Tests for the compute_gradient_and_dynamics and chain_rule functions.
"""

import pytest
import numpy as np

import oqupy


# -----------------------------------------------------------------------------
# -- Test J: Spin boson model gradient -------------------------------------------------

# Target state
target_derivative_J=np.array([[0,0.0],[0.0,1.0]])

# Initial state:
initial_state_J = np.array([[1.0,0.0],[0.0,0.0]])

# End time
t_end_J = 1.0

# Time step and number of steps
dt=0.05
num_steps=int(t_end_J/dt)
    
# Parameter at each time step
x0 = np.ones((2*num_steps,1))

# Markovian dissipation
gamma_J_1 = lambda t: 0.1 # with sigma minus
gamma_J_2 = lambda t: 0.2 # with sigma z

# Ohmic spectral density with exponential cutoff
coupling_operator_J = np.array([[0.5,0.0],[0.0,-0.5]])
alpha_J = 0.3
cutoff_J = 5.0
temperature_J = 0.2

# Result obtained with release code (made hermitian):
rho_J = np.array([[ 0.7809559 +0.j        , -0.09456333+0.16671419j],
                  [-0.09456333-0.16671419j,  0.2190441 +0.j        ]])

correlations_J = oqupy.PowerLawSD(alpha=alpha_J,
                                  zeta=1.0,
                                  cutoff=cutoff_J,
                                  cutoff_type="exponential",
                                  temperature=temperature_J,
                                  name="ohmic")
bath_J = oqupy.Bath(coupling_operator_J,
                    correlations_J,
                    name="phonon bath") 

# Parameterized Hamiltonian definition
def discrete_h_sys_J(hx):
    return 0.5*hx * oqupy.operators.sigma('x')

system_J = oqupy.ParameterizedSystem(hamiltonian=discrete_h_sys_J,
                        gammas=[gamma_J_1, gamma_J_2],
                        lindblad_operators=[lambda t: oqupy.operators.sigma("-"),
                                            lambda t: oqupy.operators.sigma("z")])

# Derivative of F(T) w.r.t. hx(0) obtained from release code
grad_params_J= [[0.00507649], [0.00534207], [0.0053693], [0.00557299], [0.00559541], 
                [0.00575529], [0.00577277], [0.0059047], [0.00591734], [0.0060279], 
                [0.0060359], [0.00612581], [0.00612941], [0.00619729], [0.00619673], 
                [0.00624089], [0.00623642], [0.00625545], [0.00624727], [0.00624009], 
                [0.00622842], [0.00619391], [0.00617894], [0.00611593], [0.00609786], 
                [0.00600489], [0.00598388], [0.00585915], [0.00583539], [0.00567655], 
                [0.00565021], [0.00545466], [0.0054259], [0.00519144], [0.00516043], 
                [0.00488701], [0.00485394], [0.00454699], [0.00451208], [0.00418564]]

def test_tempo_gradient_backend_J():
    tempo_params_J =oqupy.TempoParameters(
        dt=0.05,
        tcut=None,
        epsrel=10**(-7))
    pt = oqupy.pt_tempo_compute(
        bath_J,
        start_time=0.0,
        end_time=t_end_J,
        parameters=tempo_params_J)
    
    grad_prop,dyn = oqupy.compute_gradient_and_dynamics(system=system_J,
                                                    parameters=x0,
                                                    process_tensors=[pt],
                                                    initial_state=initial_state_J,
                                                    target_derivative=target_derivative_J.T
                                                    )

    np.testing.assert_almost_equal(dyn.states[-1], rho_J, decimal=4)

    get_props = system_J.get_propagators(dt,parameters=x0)
    get_prop_derivatives = system_J.get_propagator_derivatives(dt=dt,parameters=x0)

    grad_params = oqupy.gradient._chain_rule(adjoint_tensor=grad_prop,
                                            dprop_dparam=get_prop_derivatives,
                                            propagators=get_props,
                                            num_steps=num_steps,
                                            num_parameters=1)
    
    np.testing.assert_almost_equal(grad_params.real,grad_params_J,decimal=4)

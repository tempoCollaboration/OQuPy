# Copyright 2022 The TEMPO Collaboration
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
Frontend for computing the gradient of some objective function w.r.t. some
control parameters.
"""
from typing import Dict, List, Optional, Tuple, Callable,Union

import numpy as np
import tensornetwork as tn

from numpy import ndarray

from oqupy.contractions import compute_gradient_and_dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParameterizedSystem

def state_gradient(
        system: ParameterizedSystem,
        initial_state: ndarray,
        target_derivative: Union[Callable,ndarray],
        process_tensors: List[BaseProcessTensor],
        parameters: List[Tuple],
        time_steps: Optional[ndarray] = None,
        dynamics_only: Optional[bool] = False,
        ) -> Dict:
    """
    Compute system dynamics and gradient of an objective function Z with respect to a parameterized Hamiltonian,
    for a given set of control parameters, accounting for the interaction with an environment described by a 
    process tensor. The target state correponds to (dZ/drho_f).
    Inputs:
        system : ParameterizedSystem object to compute the dynamics
        initial_state : the initial density matrix to propagate forwards
        target_derivative : either the state to propagate backwards, or 
                        a function, which will be called with the final state and should return the 
                        state to be back-propagated.
        process_tensors : a list of process tensors [p1,p2,...] to contract with propagators and propagator 
                        derivatives. They should be ordered according to the order of application of the mpos
                        e.g. for the nth step the nth mpo of p1 is applied first, p2 second and so on.
        parameters : a list of tuples with each tuple corresponding to the value of each parameter at a given 
                        time step.
        time_steps : (Optional) a list of timesteps [0,dt,...(N-1)dt].
        dynamics_only : (Optional) when true stops the calculation after the forward propagation and returns
                        the dynamics only, otherwise does the full calculation.


    The return dictionary has the fields:
      'final state' : the final state after evolving the initial state
      'gradprop' : derivatives of Z with respect to half-step propagators  
      'gradient' : derivatives of Z with respect to parameters
                   a tuple list of floats
                   ['gradient'][i][n] ... is the derivative with respect to
                                          the i-th parameter at the n-th
                                          half-time step.
      'dynamics' : a Dynamics object (optional) 

    """
    num_steps = len(process_tensors[0])
    if time_steps is None:
        time_steps = range(num_steps) 

    grad_prop,dynamics = compute_gradient_and_dynamics(
        system=system,
        initial_state=initial_state,
        target_derivative=target_derivative,
        process_tensors=process_tensors,
        parameters=parameters,
        num_steps=num_steps,
        dynamics_only=dynamics_only)
    
    if dynamics_only:
        return_dict = {
        'final state':dynamics.states[-1],
        'dynamics':dynamics
    }
    else:
        num_parameters = len(parameters[0])
        dt = process_tensors[0].dt

        get_half_props= system.get_propagators(dt,parameters)
        get_prop_derivatives = system.get_propagator_derivatives(dt,parameters)
        

        final_derivs = _chain_rule(
            adjoint_tensor=grad_prop,
            dprop_dparam=get_prop_derivatives,
            propagators=get_half_props,
            num_steps=num_steps,
            num_parameters=num_parameters)
    
        return_dict = {
            'final state':dynamics.states[-1],
            'gradprop':grad_prop,
            'gradient':final_derivs,
            'dynamics':dynamics
        }
    
    return return_dict

def _chain_rule(
        adjoint_tensor:ndarray,
        dprop_dparam:Callable[[int], Tuple[ndarray,ndarray]],
        propagators:Callable[[int], Tuple[ndarray,ndarray]],
        num_steps:int,
        num_parameters:int):

    def combine_derivs(
            target_deriv,
            pre_prop,
            post_prop):

            target_deriv = tn.Node(target_deriv)
            pre_node=tn.Node(pre_prop)
            post_node=tn.Node(post_prop)

            target_deriv[3] ^ post_node[1] 
            target_deriv[2] ^ post_node[0] 
            target_deriv[1] ^ pre_node[1] 
            target_deriv[0] ^ pre_node[0] 

            final_node = target_deriv @ pre_node \
                            @ post_node

            tensor = final_node.tensor

            return tensor

    total_derivs = np.zeros((2*num_steps,num_parameters),dtype='complex128')

    for i in range(0,num_steps): # populating two elements each step
            
        first_half_prop,second_half_prop=propagators(i)
        
        first_half_prop_derivs,second_half_prop_derivs = dprop_dparam(i) # returns two lists containing the derivatives w.r.t. each parameter

        for j in range(0,num_parameters):

            total_derivs[2*i][j] = combine_derivs(
                            adjoint_tensor[i],
                            first_half_prop_derivs[j].T,
                            second_half_prop.T)
            
            total_derivs[2*i+1][j] = combine_derivs(
                adjoint_tensor[i],
                first_half_prop.T,
                second_half_prop_derivs[j].T)

    return total_derivs
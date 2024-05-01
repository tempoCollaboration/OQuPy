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
from typing import Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
from numpy import ndarray
import tensornetwork as tn

from oqupy.config import NpDtype, INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.contractions import _compute_dynamics_input_parse, _apply_system_superoperator,\
    _apply_derivative_pt_mpos, _get_pt_mpos, _get_pt_mpos_backprop, _get_caps, _apply_caps, _apply_pt_mpos
from oqupy.control import Control
from oqupy.dynamics import Dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParameterizedSystem
from oqupy.util import get_progress


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


def compute_gradient_and_dynamics(
        system: ParameterizedSystem,
        parameters : Optional[ndarray]=None,
        initial_state: Optional[ndarray] = None,
        target_derivative: Optional[Union[Callable,ndarray]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        process_tensors: Optional[Union[List[BaseProcessTensor],
                                       BaseProcessTensor]] = None,
        control: Optional[Control] = None,
        record_all: Optional[bool] = True,
        get_forward_and_backprop_list = False,
        dynamics_only: Optional[bool] = False,
        progress_type: Optional[Text] = None) -> Tuple[List,Dynamics]:
    """
    Compute some objective function and calculate its gradient w.r.t.
    some control parameters, accounting
    (optionally) for interaction with an environment using one or more
    process tensors.

    Parameters
    ----------
    system: Union[System, TimeDependentSystem]
        Object containing the system Hamiltonian information.
    initial_state: ndarray
        Initial system state.
    target_derivative:
        Some pure target state or derivative w.r.t. an objective functioni
    dt: float
        Length of a single time step.
    num_steps: int
        Optional number of time steps to be computed.
    start_time: float
        Optional start time offset.
    process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor]
        Optional process tensor object or list of process tensor objects.
    control: Control
        Optional control operations.
    record_all: bool
        If `false` function only computes the final state.
    subdiv_limit: int (default = config.SUBDIV_LIMIT)
        The maximum number of subdivisions used during the adaptive
        algorithm when integrating the system Liouvillian. If None
        then the Liouvillian is not integrated but sampled twice to
        to construct the system propagators at each timestep.
    liouvillian_epsrel: float (default = config.INTEGRATE_EPSREL)
        The relative error tolerance for the adaptive algorithm
        when integrating the system Liouvillian.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``silent``, ``simple``, ``bar``}. If `None` then
        the default progress type is used.

    Returns
    -------
    dynamics: Dynamics
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment).
    """

    # -- input parsing --
    parsed_parameters = _compute_dynamics_input_parse(
        False, system, initial_state, dt, num_steps, start_time,
        process_tensors, control, record_all)
    system, initial_state, dt, num_steps, start_time, \
        process_tensors, control, record_all, hs_dim = parsed_parameters

    assert target_derivative is not None, \
        'target state must be given explicitly'

    num_envs = len(process_tensors)

    # -- prepare propagators --
    propagators = system.get_propagators(dt, parameters)

    # -- prepare controls --
    def controls(step: int):
        return control.get_controls(
            step,
            dt=dt,
            start_time=start_time)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Forwardpropagation ~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- initialize computation --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg
    initial_ndarray = initial_state.reshape(hs_dim**2)
    initial_ndarray.shape = tuple([1]*num_envs+[hs_dim**2])
    current_node = tn.Node(initial_ndarray)
    current_edges = current_node[:]

    states = []
    title = "--> Compute dynamics:"
    prog_bar = get_progress(progress_type)(num_steps, title)
    prog_bar.enter()

    forwardprop_derivs_list = []
    mpo_list=[]

    for step in range(num_steps+1): 

        # -- apply pre measurement control --
        pre_measurement_control, post_measurement_control = controls(step)
 
        if pre_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, pre_measurement_control)
        
        if step == num_steps:
            break

        # -- extract current state -- update field --
        if record_all:
            caps = _get_caps(process_tensors, step)
            state_tensor = _apply_caps(current_node, current_edges, caps)
            state = state_tensor.reshape(hs_dim, hs_dim)
            states.append(state)

        prog_bar.update(step)

        # -- apply post measurement control --
        if post_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, post_measurement_control)

        forwardprop_derivs_list.append(tn.replicate_nodes([current_node])[0])

        # -- propagate one time step --
        first_half_prop, second_half_prop = propagators(step)

        pt_mpos = _get_pt_mpos(process_tensors, step)
        mpo_list.append(pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop)

    # -- extract last state --
    caps = _get_caps(process_tensors, num_steps)
    state_tensor = _apply_caps(current_node, current_edges, caps)
    final_state = state_tensor.reshape(hs_dim, hs_dim)
    states.append(final_state)

    prog_bar.update(num_steps)
    prog_bar.exit()

    # -- create dynamics object --
    if record_all:
        times = start_time + np.arange(len(states))*dt
    else:
        times = [start_time + len(states)*dt]

    if dynamics_only:
        return [], Dynamics(times=list(times),states=states)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Backpropagation ~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # -- initialize computation (except backwards) --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg

    if callable(target_derivative):
        target_derivative=target_derivative(states[-1])

    target_ndarray = target_derivative
    target_ndarray = target_ndarray.reshape(hs_dim**2)
    target_ndarray.shape = tuple([1]*num_envs+[hs_dim**2])
    current_node = tn.Node(target_ndarray) 
    current_edges = current_node[:]

    combined_deriv_list = []

    pre_measurement_control, post_measurement_control=controls(num_steps)

    if pre_measurement_control is not None:
        current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, pre_measurement_control.T)

    forwardprop_tensor = forwardprop_derivs_list[num_steps-1]  

    if get_forward_and_backprop_list:
        backprop_derivs_list.append(tn.replicate_nodes([current_node])[0])    

    pt_mpos = mpo_list[num_steps-1]
    backprop_tensor =  tn.replicate_nodes([current_node])[0]

    fwd_edges = forwardprop_tensor[:]
    deriv_forwardprop_tensor,fwd_edges = _apply_derivative_pt_mpos(forwardprop_tensor,fwd_edges,pt_mpos)

    for i,pt_mpo in enumerate(pt_mpos):      
        fwd_edges[i] ^ backprop_tensor[i] 

    deriv = deriv_forwardprop_tensor @ backprop_tensor

    combined_deriv_list.append(tn.replicate_nodes([deriv])[0])
        
    for step in reversed(range(1,num_steps)):

        # -- now the backpropagation part --
        pre_measurement_control, post_measurement_control = controls(step)
        first_half_prop, second_half_prop = propagators(step)
        pt_mpos = _get_pt_mpos_backprop(mpo_list, step)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop.T)

        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)
 
        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop.T)

        if post_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, post_measurement_control.T)

        if pre_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, pre_measurement_control.T)

        forwardprop_tensor = forwardprop_derivs_list[step-1]  

        if get_forward_and_backprop_list:
            backprop_derivs_list.append(tn.replicate_nodes([current_node])[0])
     
        backprop_tensor =  tn.replicate_nodes([current_node])[0]

        pt_mpos = mpo_list[step-1]

        fwd_edges = forwardprop_tensor[:]
        deriv_forwardprop_tensor,fwd_edges = _apply_derivative_pt_mpos(forwardprop_tensor,fwd_edges,pt_mpos)
          
        for i,pt_mpo in enumerate(pt_mpos):      
            fwd_edges[i] ^ backprop_tensor[i] 

        deriv = deriv_forwardprop_tensor @ backprop_tensor

        combined_deriv_list.append(tn.replicate_nodes([deriv])[0])

    # deriv_list is currently in the reversed order from what you'd expect, so
    # reversing the order of the list.....

    deriv_list_reversed = list(reversed(combined_deriv_list))

    # -- create dynamics object --
    if record_all:
        times = start_time + np.arange(len(states))*dt
    else:
        times = [start_time + len(states)*dt]

    dynamics_instance = Dynamics(times=list(times),states=states)

    if get_forward_and_backprop_list is False:
        forwardprop_derivs_list = None
        backprop_derivs_list = None

    return deriv_list_reversed, dynamics_instance

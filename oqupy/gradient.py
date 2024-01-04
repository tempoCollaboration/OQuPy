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

from typing import List, Optional, Text, Tuple, Union
from warnings import warn

from scipy.interpolate import interp1d

import numpy as np
from numpy import ndarray
import tensornetwork as tn

from oqupy.config import INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.control import Control
from oqupy.dynamics import GradientDynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import ParametrizedSystem
from oqupy.contractions import _compute_dynamics_input_parse
from oqupy.contractions import compute_gradient_and_dynamics
from oqupy.helpers import get_half_timesteps,get_MPO_times
from oqupy.helpers import get_propagator_intervals

# note for those reading. The arguments for the gradient function are basically
# stolen from the compute_dynamics function in contractions.py, see notes below
# because some of them are prob no longer optional.
def gradient(
        system: ParametrizedSystem,
        gradient_dynamics: Optional[GradientDynamics] = None,
        initial_state: Optional[ndarray] = None,
        target_state: Optional[ndarray] = None,
        process_tensor: Optional[Union[List[BaseProcessTensor],
                                       BaseProcessTensor]] = None,
        get_forward_backprop_list: Optional[bool] = False,
        record_all: Optional[bool] = True,
        subdiv_limit: Optional[int] = SUBDIV_LIMIT,
        liouvillian_epsrel: Optional[float] = INTEGRATE_EPSREL,
        progress_type: Optional[Text] = None) -> GradientDynamics:


    """
    compute the system dynamics as well as the gradient with respect to some
    objective function which contains a system hamiltonian, and accounting
    (optionally) for interaction with an environment using one or more process
    tensors.

    Parameters
    ----------
    system: ParamatarizedSystem
        Object containing the system Hamiltonian information, and parameters
    gradient_dynamics: GradientDynamics
    Optional: if provided, takes the recombined forwardprop and backprop and
    addes it to the specified dprop_dparam_list and produces the total gradient.
    If it is not provided then the forwardprop and backprop are performed to
    generate this object
    initial_state: ndarray
        Initial system state.
    target_state: ndarray
        Target state or derivative of target state (maybe rename variable?)
    process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor]
        Optional process tensor object or list of process tensor objects.
    get_forward_backprop_list: Bool
        save both the states stored during the forward propagation and back
        propagation to the GradientDynamics object. This is not necessary for
        obtaining the gradient through the adjoint method, and this method is
        simply provided for convenience. Also note that with this option
        disabled (the default), once the forwardprop and backprop states are
        used they are deleted. This is a significant memory saving as both
        contain a dangling internal bond leg, which is generally large.
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
    GradientDynamics:
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment), and the
        derivative of the objective function with respect to the specified
    """
    if gradient_dynamics is None:
        # -- input parsing --
        parsed_parameters = _compute_dynamics_input_parse(
            False, system, initial_state, dt, num_steps, start_time,
            process_tensor, control, record_all)
        system, initial_state, dt, num_steps, start_time, \
            process_tensors, control, record_all, hs_dim = parsed_parameters

        gradient_dynamics = compute_gradient_and_dynamics(
                        system=system,
                        initial_state=initial_state,
                        target_state=target_state,
                        dt=dt,
                        num_steps=num_steps,
                        start_time=start_time,
                        process_tensor=process_tensors,
                        control=control,
                        record_all=record_all,
                        get_forward_and_backprop_list=get_forward_backprop_list,
                        subdiv_limit=subdiv_limit,
                        liouvillian_epsrel=liouvillian_epsrel,
                        progress_type=progress_type)

    if dprop_dparam_list is not None:

        if dprop_times_list is None:
            dprop_times_list = get_half_timesteps(pt=process_tensor,start_time=start_time)

        total_derivs = _chain_rule(
                    gradient_dynamics.deriv_list,
                    dprop_dparam_list,
                    dprop_times_list,
                    start_time,
                    process_tensor,
                    system,
                    (subdiv_limit,liouvillian_epsrel))
        gradient_dynamics.total_derivs = total_derivs
    return gradient_dynamics



def _chain_rule(deriv_list: List[ndarray],
            dprop_dparam_list: List[ndarray],
            # dprop_dparam_indices = List[int],
            dprop_times_list: ndarray,
            start_time,
            process_tensor: BaseProcessTensor,
            system: Union[System,TimeDependentSystem],
            system_params = Tuple)->ndarray:
    """
    Uses chain rule to evaluate the gradient of the fidelity w.r.t. the
    control parameters, which are provided in the dprop dpram list

    NOTE: It is useful to make the dprop_times_list based off the
    helpers.get_half_timesteps method so that any floating point numbers
    are semi-dealt with. The method using scipy.interp1d should be relatively
    safe however it's there so it might as well be used.
    """
    assert len(dprop_times_list) == len(dprop_dparam_list), \
            ('dprop_dpram_list must be the same length as the number of time '
            'slices')

    MPO_times = get_MPO_times(process_tensor,start_time,inc_endtime=True)
    MPO_times = np.concatenate((np.array([0.0]),MPO_times))
    MPO_indices = np.arange(0,MPO_times.size)
    # this is more of an internal check than anything, if the code is written
    # correctly this should always be true.
    # assert indices.size == len(deriv_list), \
    #         'indices mismatched with deriv_list times'
    MPO_index_function = interp1d(MPO_times,MPO_indices,kind='zero')

    half_timestep_times = get_propagator_intervals(process_tensor,start_time)
    half_timestep_indices = np.arange(0,half_timestep_times.size)

    half_timestep_index_function = interp1d(half_timestep_times,
                                half_timestep_indices,kind='zero')

    dprop_timestep_index = half_timestep_index_function(dprop_times_list)
    dprop_timestep_index = dprop_timestep_index.astype(int)

    # test to make sure that all of the dprop_times_list time supplied actually
    # correspond to correct propagator times
    # comment out the following block of code in order to supress warning
    # ~~~~~~~~~ cut here ~~~~~~~~~
    oqupy_half_timestep_times = get_half_timesteps(
            process_tensor,start_time=start_time)

    proposed_dprop_times = oqupy_half_timestep_times[dprop_timestep_index]

    if not np.allclose(proposed_dprop_times,dprop_times_list,rtol=1e-04):
        warn('warning, you have supplied a dprop_time that is significantly '
            'different from what I calculate as your dprop_times. If you '
            'absolutely know what you are doing please suppress this message '
            'within oqupy.gradient.py or change rtol in np.allclose()' )
    # ~~~~~~~~~~ end cut ~~~~~~~~~

    total_derivs = np.zeros(dprop_times_list.size,dtype='complex128')

    propagators = system.get_propagators(
                    process_tensor.dt,
                    start_time,
                    system_params[0],
                    system_params[1])

    def combine_derivs(
                target_deriv:ndarray,
                propagator_deriv:ndarray,
                pre:ndarray,
                post:ndarray,
                pre_post_decider:bool):
        target_deriv_node = tn.Node(target_deriv)
        propagator_deriv_node = tn.Node(propagator_deriv)
        # in the following: when a pre or post is chosen, then the post / pre
        # that is not used is simply discarded

        # deriv is a post node -> extra node needed is a pre
        if pre_post_decider:
            extra_prop_node = tn.Node(pre)
            target_deriv_node[0] ^ extra_prop_node[0]
            target_deriv_node[1] ^ propagator_deriv_node[1]
            extra_prop_node[1] ^ propagator_deriv_node[0]

        # deriv is a pre node -> extra node needed is a post
        else:
            extra_prop_node = tn.Node(post)
            target_deriv_node[0] ^ propagator_deriv_node[0]
            target_deriv_node[1] ^ extra_prop_node[0]
            extra_prop_node[1] ^ propagator_deriv_node[1]

        final_node = target_deriv_node @ propagator_deriv_node \
            @ extra_prop_node

        tensor = final_node.tensor
        return tensor

    for i in range(dprop_times_list.size):
        # find the propagator index as used in contractions
        prop_index = dprop_timestep_index[i] // 2
        # decide whether it's a pre or post node
        # assuming dprop_times_list is int because it is converted to int after
        # it is created so shouldn't need to check again
        # 0/False -> Pre node, 1/True -> Post node
        pre_post_decider = dprop_timestep_index[i] % 2

        # if first or last timestep, do not include the extra propagator in
        # diagram as they are special cases where there was no propagator
        # omitted during the forward and backprop. Cleanest way to implement
        # this IMO is to set the extra propagator to the identity
        if dprop_timestep_index[i] == 0 or dprop_timestep_index[i] \
                ==  dprop_times_list.size-1:
            pre_prop = np.identity(process_tensor.hilbert_space_dimension**2)
            post_prop = np.identity(process_tensor.hilbert_space_dimension**2)

        # post node -> extra node needed comes from previous step
        elif pre_post_decider:
            pre_prop,post_prop = propagators(prop_index+1)
        # pre node -> extra node needed comes from following step
        else:
            pre_prop,post_prop = propagators(prop_index-1)

        dtarget_index = int(MPO_index_function(dprop_times_list[i]))

        total_derivs[i] = combine_derivs(
                    deriv_list[dtarget_index],
                    dprop_dparam_list[i],
                    pre_prop,
                    post_prop,
                    pre_post_decider)

    return total_derivs

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
Module for various applications involving contractions of the process tensor.
"""

from typing import List, Optional, Text, Tuple, Union

from scipy.interpolate import interp1d

import numpy as np
from numpy import ndarray
import tensornetwork as tn

from oqupy.config import NpDtype, INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.control import Control
from oqupy.dynamics import Dynamics, MeanFieldDynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import BaseSystem, System, TimeDependentSystem
from oqupy.system import MeanFieldSystem
from oqupy.operators import left_super, right_super
from oqupy.util import check_convert, check_isinstance, check_true
from oqupy.util import get_progress
from oqupy.contractions import _compute_dynamics_input_parse, compute_gradient_and_dynamics
from oqupy.helpers import get_half_timesteps,get_MPO_times

def gradient(
        system: Union[System, TimeDependentSystem],
        initial_state: Optional[ndarray] = None, # why is this optional, this is def needed
        target_state: Optional[ndarray] = None, # same again
        dprop_dparam_list: Optional[List[ndarray]] = None,
        dprop_times_list: Optional[ndarray] = None,# this one is actually optional
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        process_tensor: Optional[Union[List[BaseProcessTensor],
                                       BaseProcessTensor]] = None,
        control: Optional[Control] = None,
        record_all: Optional[bool] = True,
        subdiv_limit: Optional[int] = SUBDIV_LIMIT,
        liouvillian_epsrel: Optional[float] = INTEGRATE_EPSREL,
        progress_type: Optional[Text] = None) -> Dynamics:


    """
    compute the system dynamics for a given objective function which contains
    a system hamiltonian, and accounting (optionally) for interaction with
    an environment using one or more process tensors.

    Parameters
    ----------
    system: Union[System, TimeDependentSystem]
        Object containing the system Hamiltonian information.
    initial_state: ndarray
        Initial system state.
    dprop_times_list:
        SORTED!! list of times which the dprop_dparam_list is defined
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
                    subdiv_limit=subdiv_limit,
                    liouvillian_epsrel=liouvillian_epsrel,
                    progress_type=progress_type)

    total_derivs = _chain_rule(
                gradient_dynamics.deriv_list,
                dprop_dparam_list,
                dprop_times_list,
                start_time,
                process_tensor)
    return total_derivs



def _chain_rule(deriv_list: List[ndarray],
            dprop_dparam_list: List[ndarray],
            dprop_times_list: ndarray,
            start_time,
            process_tensor: BaseProcessTensor):
    """
    Uses chain rule to evaluate the gradient of the fidelity w.r.t. the
    control parameters, which are provided in the dprop dpram list

    NOTE: It is useful to make the dprop_times_list based off the
    helpers.get_half_timesteps method so that any floating point numbers
    are semi-dealt with cause i'm not sure how well that's gonna work
    at the moment. I'm also assuming dprop_times_list is sorted

    currently assuming each control parameter only depends on two
    system propagators (2 because of symmetric trotter splitting)
    TODO: lift this requirement
    """
    assert len(dprop_times_list) == len(dprop_dparam_list), \
                    'dprop_dpram_list must be the same length as the number of time slices'

    MPO_times = get_MPO_times(process_tensor,start_time,inc_endtime=True)
    MPO_times = np.concatenate((np.array([0.0]),MPO_times))
    indices = np.arange(0,MPO_times.size)

    index_function = interp1d(MPO_times,indices,kind='zero')

    total_derivs = np.zeros(dprop_times_list.size,dtype='complex128')

    def combine_derivs_single(target_deriv:ndarray,propagator_deriv:ndarray):
        target_deriv_node = tn.Node(target_deriv)
        propagator_deriv_node = tn.Node(propagator_deriv)
        target_deriv_node[0] ^ propagator_deriv_node[0]
        target_deriv_node[1] ^ propagator_deriv_node[1]

        # this can also be done via np.matmul(target_deriv.T,propagator_deriv)
        tensor = (target_deriv_node @ propagator_deriv_node).tensor
        return tensor

    for i in range(dprop_times_list.size):
        dtarget_index = int(index_function(dprop_times_list[i]))
        total_derivs[i] = combine_derivs_single(deriv_list[dtarget_index],dprop_dparam_list[i])

    return total_derivs

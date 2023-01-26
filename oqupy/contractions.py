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

import numpy as np
from numpy import ndarray
import tensornetwork as tn
from oqupy.system import TimeDependentSystemWithField

from oqupy.config import NpDtype, INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.control import Control
from oqupy.dynamics import Dynamics, MeanFieldDynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import BaseSystem, System, TimeDependentSystem
from oqupy.system import MeanFieldSystem
from oqupy.operators import left_super, right_super
from oqupy.util import check_convert, check_isinstance, check_true
from oqupy.util import get_progress
from oqupy.dynamics import GradientDymnamics


Indices = Union[int, slice, List[Union[int, slice]]]


# -- compute dynamics ---------------------------------------------------------

def compute_dynamics(
        system: Union[System, TimeDependentSystem],
        initial_state: Optional[ndarray] = None,
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
    Compute the system dynamics for a given system Hamiltonian, accounting
    (optionally) for interaction with an environment using one or more
    process tensors.

    Parameters
    ----------
    system: Union[System, TimeDependentSystem]
        Object containing the system Hamiltonian information.
    initial_state: ndarray
        Initial system state.
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
    num_envs = len(process_tensors)

    # -- prepare propagators --
    propagators = system.get_propagators(dt, start_time, subdiv_limit,
                                       liouvillian_epsrel)

    # -- prepare controls --
    def controls(step: int):
        return control.get_controls(
            step,
            dt=dt,
            start_time=start_time)

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

        # -- propagate one time step --
        first_half_prop, second_half_prop = propagators(step)
        pt_mpos = _get_pt_mpos(process_tensors, step)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)




        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop)

    # -- extract last state --
    caps = _get_caps(process_tensors, step)
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

    return Dynamics(times=list(times),states=states)


def compute_gradient_and_dynamics(
        system: Union[System, TimeDependentSystem],
        initial_state: Optional[ndarray] = None,
        target_state: Optional[ndarray] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        process_tensor: Optional[Union[List[BaseProcessTensor],
                                       BaseProcessTensor]] = None,
        control: Optional[Control] = None,
        record_all: Optional[bool] = True,
        get_forward_and_backprop_list = False,
        subdiv_limit: Optional[int] = SUBDIV_LIMIT,
        liouvillian_epsrel: Optional[float] = INTEGRATE_EPSREL,
        progress_type: Optional[Text] = None) -> GradientDymnamics:
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
    target_state:
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
        process_tensor, control, record_all)
    system, initial_state, dt, num_steps, start_time, \
        process_tensors, control, record_all, hs_dim = parsed_parameters

    assert target_state is not None, \
        'target state must be given explicitly'

    num_envs = len(process_tensors)

    # -- prepare propagators --
    propagators = system.get_propagators(dt, start_time, subdiv_limit,
                                       liouvillian_epsrel)

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

    forwardprop_deriv_list = []

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

        # -- propagate one time step --
        first_half_prop, second_half_prop = propagators(step)
        pt_mpos = _get_pt_mpos(process_tensors, step)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        # appropriate timeslice in diagram is here
        # store derivative node
        forwardprop_deriv_list.append(tn.replicate_nodes([current_node])[0])

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop)

    # -- extract last state --
    caps = _get_caps(process_tensors, step)
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


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~ Backpropagation ~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # -- initialize computation (except backwards) --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg

    # not sure if this is correct but to get the final cap,
    final_cap = _get_caps(process_tensors, num_steps)

    target_ndarray = target_state.reshape(hs_dim**2)
    target_ndarray.shape = tuple([1]*num_envs+[hs_dim**2])
    current_node = tn.Node(np.outer(final_cap,target_ndarray)) # might be a wire crossed or something
    current_edges = current_node[:]

    combined_deriv_list = []

    if get_forward_and_backprop_list:
        backprop_deriv_list = [tn.replicate_nodes([current_node])[0]]

        forwardprop_tensor = forwardprop_deriv_list[num_steps-1]
        backprop_tensor = backprop_deriv_list[0]

    else:
        forwardprop_tensor = forwardprop_deriv_list[num_steps-1]
        # if we're not keeping the full list, we can delete the
        # forwardprop tensor to save memory
        del forwardprop_deriv_list[num_steps-1]
        backprop_tensor = tn.replicate_nodes([current_node])[0]
        # note now backprop_deriv_list should is unnecessary

        for i in range(num_envs):
            forwardprop_tensor[i] ^ backprop_tensor[i]

        deriv = forwardprop_tensor @ backprop_tensor
        combined_deriv_list.append(deriv.tensor)


    for step in reversed(range(num_steps+1)):
        '''
        # first connect forwardprop and backprop tensors together to sum across
        # internal leg storing them as a (d_sys^2,d_sys^2) tensor
        # to save memory
        forwardprop_tensor = forwardprop_deriv_list[step-1]

        if get_forward_and_backprop_list:
            backprop_tensor =  backprop_deriv_list[num_steps - step]
        else:
            # if we're not keeping the full list, we can delete the
            # forwardprop tensor to save memory
            del forwardprop_deriv_list[num_steps-step]
            # backprop deriv list could be then deleted as it is big because
            # it contains an internal bond leg, so in principal nothing other
            # than the current tensor needs to be stored  in the backprop
            # but i haven't implemented it yet.


        for i in range(num_envs):
            forwardprop_tensor[i] ^ backprop_tensor[i]

        deriv = forwardprop_tensor @ backprop_tensor
        combined_deriv_list.append(deriv.tensor)
        '''
        # -- now the backpropagation part --

        # -- apply pre measurement control --
        pre_measurement_control, post_measurement_control = controls(step)

        if post_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, post_measurement_control.T) # possibly with a transpose

        if step == 0: # i think this is correct
            break

        # record_all not necessary for backprop as it's been done in the forwardprop

        # # -- extract current state -- update field --
        # if record_all:
        #     caps = _get_caps(process_tensors, step)
        #     state_tensor = _apply_caps(current_node, current_edges, caps)
        #     state = state_tensor.reshape(hs_dim, hs_dim)
        #     states.append(state)

        # prog_bar.update(num_steps - step) # commented

        # -- apply post measurement control --
        if pre_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, pre_measurement_control.T)

        # -- propagate one time step --
        # we're propagating backwards so we're actually using the propagators
        # from the previous timestep, hence -1 in next line
        first_half_prop, second_half_prop = propagators(step-1)
        pt_mpos = _get_pt_mpos_backprop(process_tensors, step-1)

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, second_half_prop.T)
        current_node, current_edges = _apply_pt_mpos(
            current_node, current_edges, pt_mpos)

        # appropriate timeslice in diagram is here
        # # store derivative node
        # if get_forward_and_backprop_list:
        #     backprop_deriv_list.append(tn.replicate_nodes([current_node])[0])
        # else:
        #     backprop_tensor = tn.replicate_nodes([current_node])[0]
        # first connect forwardprop and backprop tensors together to sum across
        # internal leg storing them as a (d_sys^2,d_sys^2) tensor
        # to save memory
        forwardprop_tensor = forwardprop_deriv_list[step-1]

        if get_forward_and_backprop_list:
            backprop_deriv_list.append(tn.replicate_nodes([current_node])[0])
            backprop_tensor =  tn.replicate_nodes([current_node])[0]
        else:
            # if we're not keeping the full list, we can delete the
            # forwardprop tensor to save memory
            del forwardprop_deriv_list[num_steps-step]
            backprop_tensor =  tn.replicate_nodes([current_node])[0]



        for i in range(num_envs):
            forwardprop_tensor[i] ^ backprop_tensor[i]

        deriv = forwardprop_tensor @ backprop_tensor
        combined_deriv_list.append(deriv.tensor) 

        current_node, current_edges = _apply_system_superoperator(
            current_node, current_edges, first_half_prop.T)


    # -- create dynamics object --
    if record_all:
        times = start_time + np.arange(len(states))*dt
    else:
        times = [start_time + len(states)*dt]

    if get_forward_and_backprop_list == False:
        forwardprop_deriv_list = None
        backprop_deriv_list = None


    return GradientDymnamics(times=list(times),
            states=states,
            forwardprop_deriv_list=forwardprop_deriv_list,
            backprop_deriv_list=backprop_deriv_list,
            deriv_list=combined_deriv_list)




def compute_dynamics_with_field(
        mean_field_system: MeanFieldSystem,
        initial_field: complex,
        process_tensor_list: Optional[List[
            Union[BaseProcessTensor, List[BaseProcessTensor]] ]] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        initial_state_list: Optional[List[ndarray]] = None,
        start_time: Optional[float] = 0.0,
        control_list: Optional[List[Control]] = None,
        record_all: Optional[bool] = True,
        subdiv_limit: Optional[int] = SUBDIV_LIMIT,
        liouvillian_epsrel: Optional[float] = INTEGRATE_EPSREL,
        progress_type: Optional[Text] = None) -> MeanFieldDynamics:
    """
    Compute each system and field dynamics for a MeanFieldSystem
    with (optional) process tensors for each system to account for their
    interaction with their environment.

    Parameters
    ----------
    mean_field_system: MeanFieldSystem
        The `MeanFieldSystem` representing the collection of time-dependent
        systems and coherent field.
    initial_field: complex
        The initial field value.
    process_tensor_list: List[Union[List[BaseProcessTensor],BaseProcessTensor]]
        Process tensors for each system. Each element can be a BaseProcessTensor
        or a list of BaseProcessTensors for the respective system.
    dt: float
        Length of a single time step.
    initial_state_list: List[ndarray]
        List of initial density matrices, one for each system in the
        mean-field system.
    start_time: float (default = 0.0)
        Optional start time offset.
    num_steps: int
        Optional number of time steps to be computed.
    control_list: List[Control]
        Optional list of control operations.
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
    dynamics_with_field: MeanFieldDynamics
        The instance of `MeanFieldDynamics` describing each system
        dynamics and the field dynamics accounting for the interaction with
        the environment.
    """

    # initialize objects as lists where necessary
    initial_field = check_convert(initial_field, complex, "initial_field")

    assert isinstance(mean_field_system, MeanFieldSystem), \
            "Argument 'mean_field_system' must be an instance of " \
            "MeanFieldSystem."

    number_of_systems = len(mean_field_system.system_list)
    if initial_state_list is None:
        initial_state_list = [None] * number_of_systems

    if control_list is None:
        control_list = [None] * number_of_systems

    if process_tensor_list is None:
        process_tensor_list = [None] * number_of_systems


    # -- input parsing --
    # check that lengths of lists provided are consistent
    assert number_of_systems == len(initial_state_list) \
            == len(control_list),\
                f"The length of initial_state_list "\
                f"({len(initial_state_list)}) and control_list "\
                f"({len(control_list)}) must match the number of "\
                f"systems ({len(mean_field_system.system_list)}) in "\
                f"mean_field_system."

    assert isinstance(process_tensor_list, list), "process_tensor_list must "\
            "be a (possibly nested) list of BaseProcessTensor objects."
    assert len(process_tensor_list) == len(mean_field_system.system_list), \
            f"The length of process_tensor_list ({len(process_tensor_list)}) "\
            "must match the number of systems "\
            f"({len(mean_field_system.system_list)}) in mean_field_system."

    # list of tuples in the order: system, initial_state, dt, num_steps,
    # start_time, process_tensors, control, record_all, hs_dim
    parsed_parameters_tuple_list =  \
            [_compute_dynamics_input_parse(True, system, initial_state, dt,
                num_steps, start_time, process_tensor, control, record_all)
                for system, initial_state, process_tensor, control
                in zip(mean_field_system.system_list, initial_state_list,
                    process_tensor_list, control_list)]

    # parameter names returned by _compute_dynamics_input_parse()
    parsed_parameter_names = ["system", "initial_state", "dt", "num_steps",
                              "start_time", "process_tensors", "control",
                              "record_all", "hs_dim"]

    # create dictionary for parsed parameters with each key being a parameter
    # corresponding to a relevant list as its value
    parsed_parameters_dict = {}
    for i, parsed_parameter_name in enumerate(parsed_parameter_names):
        parsed_parameters_dict[parsed_parameter_name] = []
        for parsed_parameter_tuple in parsed_parameters_tuple_list:
            parsed_parameters_dict[parsed_parameter_name].append(
                    parsed_parameter_tuple[i])

    num_steps = parsed_parameters_dict["num_steps"][0]
    dt = parsed_parameters_dict["dt"][0]
    record_all = parsed_parameters_dict["record_all"][0]
    num_envs_list = [len(process_tensors) for process_tensors
                     in parsed_parameters_dict["process_tensors"]]

    propagators_list = [system.get_propagators(dt, start_time, subdiv_limit,
                                         liouvillian_epsrel)
                        for system in parsed_parameters_dict["system"]]

    # -- prepare compute field --
    def compute_field(t: float, dt: float, state_list: List[ndarray],
            field: complex, next_state_list: List[ndarray]):

        rk1 = mean_field_system.field_eom(t, state_list, field)
        rk2 = mean_field_system.field_eom(t + dt, next_state_list,
                                          field + rk1 * dt)
        return field + dt * (rk1 + rk2) / 2

    # -- prepare controls --
    def prepare_controls(step: int, control:Control):
        return control.get_controls(
            step,
            dt=dt,
            start_time=start_time)

    # -- initialize computation --
    #
    #  Initial state including the bond legs to the environments with:
    #    edges 0, 1, .., num_envs-1    are the bond legs of the environments
    #    edge  -1                      is the state leg

    nodes_and_edges_list = [] # list of tuples (current_nodes, current_edges)

    for initial_state, hs_dim, num_envs \
        in zip(parsed_parameters_dict["initial_state"],
               parsed_parameters_dict["hs_dim"],
               num_envs_list):
        initial_ndarray = initial_state.reshape(hs_dim**2)
        initial_ndarray.shape = tuple([1]*num_envs+[hs_dim**2])
        current_node = tn.Node(initial_ndarray)
        current_edges = current_node[:]

        nodes_and_edges_list.append((current_node, current_edges))

    # initialize list to store system states and field at each time step
    system_states_list = []
    field_list = []
    title = "--> Compute dynamics with field:"
    prog_bar = get_progress(progress_type)(num_steps, title)
    prog_bar.enter()

    for step in range(num_steps+1):

        # -- calculate time reached --
        t = start_time + step * dt

        # -- apply pre measurement control --
        controls_tuple_list = [prepare_controls(step, control)
                               for control in parsed_parameters_dict["control"]]

        for (current_node, current_edges), \
        (pre_measurement_control, post_measurement_control) in \
        zip(nodes_and_edges_list, controls_tuple_list):
            if pre_measurement_control is not None:
                current_node, current_edges = _apply_system_superoperator(
                        current_node, current_edges, pre_measurement_control)

        if step == num_steps:
            break

        # -- extract current states -- update field --
        caps_list = [_get_caps(process_tensors, step) for process_tensors
                     in parsed_parameters_dict["process_tensors"]]

        state_tensor_list = [_apply_caps(current_node, current_edges, caps)
                             for (current_node, current_edges), caps
                             in zip(nodes_and_edges_list, caps_list)]

        state_list = [state_tensor.reshape((hs_dim, hs_dim))
                      for state_tensor, hs_dim
                      in zip(state_tensor_list,
                             parsed_parameters_dict["hs_dim"])]

        if step == 0:
            field = initial_field
        else:
            field = compute_field(t, dt, previous_state_list, field, state_list)
        previous_state_list = state_list
        if record_all:
            system_states_list.append(state_list)
            field_list.append(field)

        prog_bar.update(step)

        # -- apply post measurement control --
        for (current_node, current_edges), \
                (pre_measurement_control, post_measurement_control) in \
            zip(nodes_and_edges_list, controls_tuple_list):
            if post_measurement_control is not None:
                current_node, current_edges = _apply_system_superoperator(
                        current_node, current_edges, post_measurement_control)

        # -- propagate one time step --
        propagator_tuples_list = [propagators(step, field,
                                mean_field_system.field_eom(t, state_list,
                                                            field))
                                for propagators in propagators_list]

        pt_mpos_list = [_get_pt_mpos(process_tensors, step) for process_tensors
                        in parsed_parameters_dict["process_tensors"]]


        # first half propagator
        nodes_and_edges_list = [_apply_system_superoperator(
                                    current_node, current_edges,
                                    first_half_prop)
                                    for (current_node, current_edges), \
                                        (first_half_prop, second_half_prop)
                                    in zip(nodes_and_edges_list,
                                           propagator_tuples_list)]

        # PT-MPO
        nodes_and_edges_list = [_apply_pt_mpos(current_node, current_edges,
                                               pt_mpos)
                                for (current_node, current_edges), pt_mpos
                                in zip(nodes_and_edges_list, pt_mpos_list)]

        # second half propagator
        nodes_and_edges_list = [_apply_system_superoperator(
                                    current_node, current_edges,
                                    second_half_prop)
                                for (current_node, current_edges), \
                                        (first_half_prop, second_half_prop)
                                in zip(nodes_and_edges_list,
                                       propagator_tuples_list)]

    # -- extract last states --
    caps_list = [_get_caps(process_tensors, step) for process_tensors
                 in parsed_parameters_dict["process_tensors"]]

    state_tensor_list = [_apply_caps(current_node, current_edges, caps)
                         for (current_node, current_edges), caps
                         in zip(nodes_and_edges_list, caps_list)]

    final_state_list = [state_tensor.reshape(hs_dim, hs_dim)
                        for state_tensor, hs_dim
                        in zip(state_tensor_list,
                               parsed_parameters_dict["hs_dim"])]

    system_states_list.append(final_state_list)

    final_field = compute_field(t, dt, previous_state_list, field,
                                final_state_list)
    field_list.append(final_field)

    prog_bar.update(num_steps)
    prog_bar.exit()

    # -- create dynamics object --
    if record_all:
        times = start_time + np.arange(len(system_states_list))*dt
    else:
        times = [start_time + len(system_states_list)*dt]

    return MeanFieldDynamics(
                times=list(times), system_states_list=system_states_list,
                fields=field_list)

def _compute_dynamics_input_parse(
        with_field, system, initial_state, dt, num_steps, start_time,
        process_tensor, control, record_all) -> tuple:

    if with_field:
        check_isinstance(system, TimeDependentSystemWithField, "system")
    else:
        check_isinstance(system, (System, TimeDependentSystem), "system")

    hs_dim = system.dimension

    if process_tensor is None:
        process_tensor = []
    check_isinstance(
        process_tensor, (BaseProcessTensor, list), "process_tensor")

    if isinstance(process_tensor, BaseProcessTensor):
        process_tensors = [process_tensor]
    elif isinstance(process_tensor, list):
        process_tensors = process_tensor

    if control is None:
        control = Control(hs_dim)
    check_isinstance(control, Control, "control")

    start_time = check_convert(start_time, float, "start_time")

    if initial_state is None:
        raise ValueError("An initial state must be specified.")
    check_isinstance(initial_state, ndarray, "initial_state")
    check_true(
        initial_state.shape == (hs_dim, hs_dim),
        "Initial sate must be a square matrix of " \
            + f"dimension {hs_dim}x{hs_dim}.")

    max_steps = []
    for pt in process_tensors:
        check_isinstance(
           pt, BaseProcessTensor, "pt",
           "One of the elements in `process_tensor` is not of type " \
               + "`{BaseProcessTensor.__name__}`.")
        if pt.get_initial_tensor() is not None:
            raise NotImplementedError()
        check_true(
            hs_dim == pt.hilbert_space_dimension,
            "All process tensor must have the same Hilbert "\
                    "space dimension as the system.")
        if pt.dt is not None:
            if dt is None:
                dt = pt.dt
            else:
                check_true(
                    pt.dt == dt,
                    "All process tensors must have the same "\
                            "timestep length.")
        max_steps.append(pt.max_step)
    max_step = np.min(max_steps+[np.inf])

    if dt is None:
        raise ValueError(
            "No timestep length has been specified. Please set `dt`.")

    if num_steps is not None:
        num_steps = check_convert(num_steps, int, "num_steps")
        check_true(
            num_steps <= max_step,
            "Variable `num_steps` is larger than the shortest "\
                    "process tensor!")
    else:
        check_true(
            max_step < np.inf,
            "Variable `num_steps` must be specified because all "\
                    "process tensors involved are infinite.")
        num_steps = int(max_step)

    parameters = (system, initial_state, dt, num_steps, start_time, \
                  process_tensors, control, record_all, hs_dim)
    return parameters


def _get_caps(process_tensors: List[BaseProcessTensor], step: int):
    """ToDo """
    caps = []
    for i in range(len(process_tensors)):
        try:
            cap = process_tensors[i].get_cap_tensor(step)
            caps.append(cap)
        except Exception as e:
            raise ValueError("There are either no cap tensors in "\
                    +f"process tensor {i} or process tensor {i} is "\
                    +"not long enough") from e
        if cap is None:
            raise ValueError(f"Process tensor {i} has no cap tensor "\
                +f"for step {step}.")
    return caps


def _get_pt_mpos(process_tensors: List[BaseProcessTensor], step: int):
    """ToDo """
    pt_mpos = []
    for i in range(len(process_tensors)):
        pt_mpo = process_tensors[i].get_mpo_tensor(step)
        pt_mpos.append(pt_mpo)
    return pt_mpos

def _get_pt_mpos_backprop(process_tensors: List[BaseProcessTensor], step: int):
    """same as above but swaps the system legs and internal bond legs
    before returning MPOs.

       [forwardprop]
         [1]
          |       [3]
          |      /
    |---------| /
    |         |/
    |         |\       propagate
    |---------| \         ^
          |      \        |
          |       [2]
         [0]

          |
          |
         \ /
          v
       [backprop]
         [0]
          |       [3]
          |      /
    |---------| /
    |         |/
    |         |\       propagate
    |---------| \         ^
          |      \        |
          |       [2]
         [1]
    """
    pt_mpos = []
    for i in range(len(process_tensors)):
        pt_mpo = process_tensors[i].get_mpo_tensor(step)
        # now swap axes so propagating upwards on the PT diagram propagates
        # *backwards* in time
        pt_mpo = np.swapaxes(pt_mpo,0,1) # internal bond legs
        pt_mpo = np.swapaxes(pt_mpo,2,3) # system propagator legs
        pt_mpos.append(pt_mpo)
    return pt_mpos


def _apply_system_superoperator(current_node, current_edges, sup_op):
    """ToDo """
    sup_op_node = tn.Node(sup_op.T)
    current_edges[-1] ^ sup_op_node[0]
    new_sys_edge = sup_op_node[1]
    current_node = current_node @ sup_op_node
    current_edges[-1] = new_sys_edge
    return current_node, current_edges


def _apply_caps(current_node, current_edges, caps):
    """ToDo """
    node_dict, edge_dict = tn.copy([current_node])
    for current_edge, cap in zip(current_edges[:-1], caps):
        cap_node = tn.Node(cap)
        edge_dict[current_edge] ^ cap_node[0]
        node_dict[current_node] = node_dict[current_node] @ cap_node
    state_node = node_dict[current_node]
    return state_node.get_tensor()


def _apply_pt_mpos(current_node, current_edges, pt_mpos):
    """ToDo """
    for i, pt_mpo in enumerate(pt_mpos):
        if pt_mpo is None:
            continue
        pt_mpo_node = tn.Node(pt_mpo)
        new_bond_edge = pt_mpo_node[1]
        new_sys_edge = pt_mpo_node[3]
        current_edges[i] ^ pt_mpo_node[0]
        current_edges[-1] ^ pt_mpo_node[2]
        current_node = current_node @ pt_mpo_node
        current_edges[i] = new_bond_edge
        current_edges[-1] = new_sys_edge
    return current_node, current_edges



# -- compute correlations ------------------------------------------------

def compute_correlations(
        system: BaseSystem,
        process_tensor: BaseProcessTensor,
        operator_a: ndarray,
        operator_b: ndarray,
        times_a: Union[Indices, float, Tuple[float, float]],
        times_b: Union[Indices, float, Tuple[float, float]],
        time_order: Optional[Text] = "ordered",
        initial_state: Optional[ndarray] = None,
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
        progress_type: Text = None,
    ) -> Tuple[ndarray, ndarray, ndarray]:
    r"""
    Compute system correlations for a given system Hamiltonian.

    Times may be specified with indices, a single float, or a pair of floats
    specifying the start and end time. Indices may be integers, slices, or
    lists of integers and slices.

    Parameters
    ----------
    system: BaseSystem
        Object containing the system Hamiltonian.
    process_tensor: BaseProcessTensor
        A process tensor object.
    operator_a: ndarray
        System operator :math:`\hat{A}`.
    operator_b: ndarray
        System operator :math:`\hat{B}`.
    times_a: Union[Indices, float, Tuple[float, float]]
        Time(s) :math:`t_A`.
    times_b: Union[Indices, float, Tuple[float, float]]
        Time(s) :math:`t_B`.
    time_order: str (default = ``'ordered'``)
        Which two time correlations to compute. Types are:
        {``'ordered'``, ``'anti'``, ``'full'``}.
    initial_state: ndarray (default = None)
        Initial system state.
    start_time: float (default = 0.0)
        Initial time.
    dt: float (default = None)
        Time step size.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``'silent'``, ``'simple'``, ``'bar'``}. If `None` then
        the default progress type is used.

    Returns
    -------
    times_a: ndarray
        The :math:`N` times :math:`t^A_n`.
    times_b: ndarray
        The :math:`M` times :math:`t^B_m`.
    correlations: ndarray
        The :math:`N \times M` correlations
        :math:`\langle B(t^B_m) A(t^A_n) \rangle`.
        Entries that are outside the scope specified in `time_order` are set to
        be `NaN + NaN j`.
    """
    # -- input parsing --
    assert isinstance(system, BaseSystem)
    assert isinstance(process_tensor, BaseProcessTensor)
    dim = system.dimension
    assert process_tensor.hilbert_space_dimension == dim
    assert operator_a.shape == (dim, dim)
    assert operator_b.shape == (dim, dim)
    assert isinstance(start_time, float)

    if dt is None:
        assert process_tensor.dt is not None, \
            "It is necessary to specify time step `dt` because the given " \
             + "tensor has none."
        dt_ = process_tensor.dt
    else:
        if (process_tensor.dt is not None) and (process_tensor.dt != dt):
            UserWarning("Specified time step `dt` does not match `dt` " \
                + "stored in the given process tensor " \
                + f"({dt}!={process_tensor.dt}). " \
                + "Using specified `dt`. " \
                + "Don't specify `dt` to use the time step stored in the " \
                + "process tensor.")
        dt_ = dt

    max_step = len(process_tensor)
    times_a_ = _parse_times(times_a, max_step, dt_, start_time)
    times_b_ = _parse_times(times_b, max_step, dt_, start_time)

    ret_times_a = start_time + dt_ * times_a_
    ret_times_b = start_time + dt_ * times_b_
    ret_correlations = np.empty((len(times_a_), len(times_b_)), dtype=NpDtype)
    ret_correlations[:] = np.NaN + 1.0j*np.NaN

    parameters = {
        "system": system,
        "process_tensor": process_tensor,
        "initial_state": initial_state,
        "start_time": start_time,
        }

    schedule = _schedule_correlation_computations(
        times_a=times_a_,
        times_b=times_b_,
        time_order=time_order)

    progress = get_progress(progress_type)
    num_steps = len(schedule)
    title = "--> Compute correlations:"
    with progress(num_steps, title) as prog_bar:
        prog_bar.update(0)
        for i, (indices_a, indices_b, anti_time_ordered) in enumerate(schedule):
            if anti_time_ordered:
                first_time = int(times_b_[indices_b])
                first_operator = operator_b
                last_times= times_a_[indices_a]
                last_operator = operator_a
            else:
                first_time = int(times_a_[indices_a])
                first_operator = operator_a
                last_times= times_b_[indices_b]
                last_operator = operator_b

            corr = _compute_ordered_correlations(
                first_time = first_time,
                first_operator = first_operator,
                last_times = last_times,
                last_operator = last_operator,
                anti_time_ordered = anti_time_ordered,
                **parameters)
            ret_correlations[indices_a, indices_b] = corr
            prog_bar.update(i+1)
    return ret_times_a, ret_times_b, ret_correlations

def _compute_ordered_correlations(
        system: BaseSystem,
        process_tensor: BaseProcessTensor,
        first_operator: ndarray,
        last_operator: ndarray,
        first_time: int,
        last_times: ndarray,
        anti_time_ordered: Optional[bool] = False,
        initial_state: Optional[ndarray] = None,
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
    ) -> Tuple[ndarray]:
    """
    Compute ordered system correlations for a given system Hamiltonian.
    """

    for last_time in last_times:
        assert first_time <= last_time

    if anti_time_ordered:
        first_super_operator = right_super(first_operator)
    else:
        first_super_operator = left_super(first_operator)

    max_step = last_times.max()
    dim = system.dimension
    control = Control(dim)
    control.add_single(first_time, first_super_operator)

    dynamics = compute_dynamics(
        system=system,
        process_tensor=process_tensor,
        control=control,
        start_time=start_time,
        initial_state=initial_state,
        dt=dt,
        num_steps=max_step,
        progress_type='silent')
    _, corr = dynamics.expectations(last_operator)
    ret_correlations = corr[last_times]
    return ret_correlations

def _schedule_correlation_computations(
        times_a: ndarray,
        times_b: ndarray,
        time_order: Text,
    ) -> List[Tuple[ndarray,ndarray,bool]]:
    """Figure out in which order to calculate two time correlations."""
    if time_order == "ordered":
        ordered = True
        anti_ordered = False
    elif time_order == "anti":
        ordered = False
        anti_ordered = True
    elif time_order == "full":
        ordered = True
        anti_ordered = True
    else:
        raise ValueError("Parameter `time_order` must be either " \
            + "``ordered``, ``anti``, or ``full``.")

    schedule = []
    if ordered:
        for i_a, time_a in enumerate(times_a):
            later_i_bs = []
            for i_b, time_b in enumerate(times_b):
                if time_a <= time_b:
                    later_i_bs.append(i_b)
            if len(later_i_bs)>0:
                schedule.append((np.array(i_a), np.array(later_i_bs), False))
    if anti_ordered:
        for i_b, time_b in enumerate(times_b):
            later_i_as = []
            for i_a, time_a in enumerate(times_a):
                if time_b < time_a:
                    later_i_as.append(i_a)
            if len(later_i_as)>0:
                schedule.append((np.array(later_i_as), np.array(i_b), True))

    return schedule

def _parse_times(times, max_step, dt, start_time):
    """Input parsing of specified time steps or time interval. """
    if isinstance(times, int):
        if times < 0 or times > max_step:
            raise IndexError("Specified time is out of bound.")
        ret_times = np.array([times])
    elif isinstance(times, (slice, list)):
        try:
            ret_times = np.arange(max_step + 1)[times]
        except Exception as e:
            raise IndexError("Specified times are invalid or out of bound.") \
                from e
    elif isinstance(times, float):
        index = int(np.round((times-start_time)/dt))
        if index < 0 or index > max_step:
            raise IndexError("Specified time is out of bound.")
        ret_times = np.array([index])
    elif isinstance(times, tuple):
        assert len(times) == 2 and isinstance(times[0], float) \
            and isinstance(times[1], float), \
            "When specifying time interval, start and end must be floats."
        index_start = int(np.round((times[0] - start_time) / dt))
        if index_start < 0 or index_start > max_step:
            raise IndexError("Specified start time is out of bound.")
        index_end = int(np.round((times[1] - start_time) / dt))
        if index_end < 0 or index_end > max_step:
            raise IndexError("Specified end time is out of bound.")
        direction = 1 if index_start <= index_end else -1
        ret_times = np.arange(max_step + 1)[index_start:index_end:direction]
    else:
        raise TypeError("Parameters `times_a` and `times_b` must be either " \
            + "int, slice, list, or tuple.")
    return ret_times

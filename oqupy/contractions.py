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
from scipy.linalg import expm
from scipy import integrate

from oqupy.config import NpDtype, INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.control import Control
from oqupy.dynamics import Dynamics, DynamicsWithField
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import BaseSystem, System, TimeDependentSystem
from oqupy.system import TimeDependentSystemWithField
from oqupy.operators import left_super, right_super
from oqupy.util import check_convert, check_isinstance, check_true
from oqupy.util import get_progress


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
        progress_type: Optional[Text] = None) -> Dynamics:
    """
    Compute the system dynamics for a given system Hamiltonian.

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
    dynamics = _compute_dynamics_all(
        False, system, None, initial_state, dt, num_steps, start_time,
        process_tensor, control, record_all, epsrel=None, subdiv_limit=None,
        progress_type=progress_type)
    return dynamics


def compute_dynamics_with_field(
        system: TimeDependentSystemWithField,
        initial_field: complex,
        initial_state: Optional[ndarray] = None,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        process_tensor: Optional[Union[List[BaseProcessTensor],
                                       BaseProcessTensor]] = None,
        control: Optional[Control] = None,
        record_all: Optional[bool] = True,
        epsrel: Optional[float] = INTEGRATE_EPSREL,
        subdiv_limit: Optional[int] = SUBDIV_LIMIT,
        progress_type: Optional[Text] = None) -> DynamicsWithField:
    """
    Compute the system and field dynamics for a given system Hamiltonian and
    field equation of motion.

    Parameters
    ----------
    system: TimeDependentSystemWithField
        Object containing the system Hamiltonian and field equation of
        motion.
    initial_field: complex
        Initial field value.
    initial_state: ndarray
        Initial system state.
    dt: float
        Length of a single time step.
    start_time: float
        Optional start time offset.
    num_steps: int
        Optional number of time steps to be computed.
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
    epsrel: float (default = config.INTEGRATE_EPSREL)
        The relative error tolerance for the adaptive algorithm
        when integrating the system Liouvillian.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``silent``, ``simple``, ``bar``}. If `None` then
        the default progress type is used.

    Returns
    -------
    dynamics_with_field: DynamicsWithField
        The system and field dynamics for the given system Hamiltonian and
        field equation of motion (accounting for the interaction with the
        environment).
    """
    initial_field = check_convert(initial_field, complex, "initial_field")
    dynamics_with_field = _compute_dynamics_all(
        True, system, initial_field, initial_state, dt, num_steps, start_time,
        process_tensor, control, record_all, epsrel, subdiv_limit,
        progress_type=progress_type)
    return dynamics_with_field


def _compute_dynamics_all(
        with_field: bool,
        system: BaseSystem,
        initial_field: float,
        initial_state: ndarray,
        dt: float,
        num_steps: int,
        start_time: float,
        process_tensor: Union[List[BaseProcessTensor], BaseProcessTensor],
        control: Control,
        record_all: bool,
        epsrel: float,
        subdiv_limit: int,
        progress_type: Text) -> Union[Dynamics, DynamicsWithField]:
    """Compute system and (optionally) field dynamics accounting for the
    interaction with the environment using the process tensor."""

    # -- input parsing --
    parsed_parameters = _compute_dynamics_input_parse(
        with_field, system, initial_state, dt, num_steps, start_time,
        process_tensor, control, record_all)
    system, initial_state, dt, num_steps, start_time, \
        process_tensors, control, record_all, hs_dim = parsed_parameters
    num_envs = len(process_tensors)

    # -- prepare propagators --
    propagators = _get_propagators(system, dt, start_time, epsrel, subdiv_limit)

    # -- prepare compute field --
    if with_field:
        def compute_field(step:int, state: ndarray, field: complex,
                next_state: Optional[ndarray] = None):
            t = start_time + step * dt
            rk1 = system.field_eom(t, state, field)
            if next_state is None:
                return rk1 * dt
            rk2 = system.field_eom(t + dt, next_state, field + rk1 * dt)
            return field + dt * (rk1 + rk2) / 2

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
    if with_field:
        fields = []

    if with_field:
        title = "--> Compute dynamics with field:"
    else:
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
        if record_all or with_field:
            caps = _get_caps(process_tensors, step)
            state_tensor = _apply_caps(current_node, current_edges, caps)
            state = state_tensor.reshape(hs_dim, hs_dim)
        if with_field:
            if step == 0:
                field = initial_field
            else:
                field = compute_field(step, previous_state, field, state)
            previous_state = state
        if record_all:
            states.append(state)
            if with_field:
                fields.append(field)

        prog_bar.update(step)

        # -- apply post measurement control --
        if post_measurement_control is not None:
            current_node, current_edges = _apply_system_superoperator(
                current_node, current_edges, post_measurement_control)

        # -- propagate one time step --
        if with_field:
            first_half_prop, second_half_prop = propagators(step, state, field)
        else:
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
    if with_field:
        final_field = compute_field(step, previous_state, field, final_state)
        fields.append(final_field)

    prog_bar.update(num_steps)
    prog_bar.exit()

    # -- create dynamics object --
    if record_all:
        times = start_time + np.arange(len(states))*dt
    else:
        times = [start_time + len(states)*dt]

    if with_field:
        return DynamicsWithField(times=list(times),states=states, fields=fields)

    return Dynamics(times=list(times),states=states)

def _get_propagators(
        system, dt, start_time, epsrel, subdiv_limit):
    """Prepare propagators according to system type and subdiv_limit"""
    if isinstance(system, System):
        first_step = expm(system.liouvillian()*dt/2.0)
        second_step = expm(system.liouvillian()*dt/2.0)
        def propagators(step: int):
            return first_step, second_step
    elif isinstance(system, TimeDependentSystem):
        def propagators(step: int):
            t = start_time + step * dt
            first_step = expm(system.liouvillian(t+dt/4.0)*dt/2.0)
            second_step = expm(system.liouvillian(t+dt*3.0/4.0)*dt/2.0)
            return first_step, second_step
    elif isinstance(system, TimeDependentSystemWithField):
        if subdiv_limit is None:
            def propagators(step: int, state: ndarray, field: complex):
                t = start_time + step * dt
                first_step = expm(system.liouvillian(t, t+dt/4.0, state,
                    field)*dt/2.0)
                second_step = expm(system.liouvillian(t, t+dt*3.0/4.0, state,
                    field)*dt/2.0)
                return first_step, second_step
        else:
            def propagators(step: int, state: ndarray, field: complex):
                t = start_time + step * dt
                liouvillian = lambda tau: system.liouvillian(t, tau, state,
                        field)
                first_step = expm(integrate.quad_vec(liouvillian,
                                                     a=t,
                                                     b=t+dt/2.0,
                                                     epsrel=epsrel,
                                                     limit=subdiv_limit)[0])
                second_step = expm(integrate.quad_vec(liouvillian,
                                                      a=t+dt/2.0,
                                                      b=t+dt,
                                                      epsrel=epsrel,
                                                      limit=subdiv_limit)[0])
                return first_step, second_step
    else:
        raise NotImplementedError("System type {} unknown".format(
            system.__name__))
    return propagators


def _compute_dynamics_input_parse(
        with_field, system, initial_state, dt, num_steps, start_time,
        process_tensor, control, record_all) -> tuple:
    if with_field:
        check_isinstance(
            system, TimeDependentSystemWithField, "system")
    else:
        check_isinstance(
            system, (System, TimeDependentSystem), "system")
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
            "All process tensor must have the same hilbert space dimension " \
                + "as the system.")
        if pt.dt is not None:
            if dt is None:
                dt = pt.dt
            else:
                check_true(
                    pt.dt == dt,
                    "All process tensors must have the same timestep length.")
        max_steps.append(pt.max_step)
    max_step = np.min(max_steps+[np.inf])

    if dt is None:
        raise ValueError(
            "No timestep length has been specified. Please set `dt`.")

    if num_steps is not None:
        num_steps = check_convert(num_steps, int, "num_steps")
        check_true(
            num_steps <= max_step,
            "Variable `num_steps` is larger than the shortest process tensor!")
    else:
        check_true(
            max_step < np.inf,
            "Variable `num_steps` must be specified because all process " \
                + "tensors involved are infinite.")
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



# -- compute correlations -----------------------------------------------------

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

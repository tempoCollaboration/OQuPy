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

from typing import Callable, List, Optional, Text, Tuple, Union

import numpy as np
from numpy import ndarray
import tensornetwork as tn
from scipy.linalg import expm

from oqupy import util
from oqupy.config import NpDtype
from oqupy.control import Control
from oqupy.dynamics import Dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import BaseSystem
from oqupy.operators import identity, left_super, right_super
from oqupy.util import get_progress


Indices = Union[int, slice, List[Union[int, slice]]]

def compute_dynamics(
        system: BaseSystem,
        process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor],
        initial_state: Optional[ndarray] = None,
        control: Optional[Control] = None,
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> Dynamics:
    """
    Compute the system dynamics for a given system Hamiltonian.

    Parameters
    ----------
    system: BaseSystem
        Object containing the system Hamiltonian information.
    process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor]
        A process tensor object or list of process tensor objects.
    initial_state: ndarray
        Initial system state.
    control: Control
        Optional control operations.
    start_time: float
        Optional start time offset.
    num_steps: int
        Optional number of time steps to be computed.
    record_all: bool
        If `false` function also computes the final state.

    Returns
    -------
    dynamics: Dynamics
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment).
    """
    # -- input parsing --
    assert isinstance(system, BaseSystem), \
        "Parameter `system` is not of type `oqupy.BaseSystem`."

    if not isinstance(process_tensor, List):
        process_tensors = [process_tensor]
    else:
        process_tensors = process_tensor

    for pt in process_tensors:
        assert isinstance(pt, BaseProcessTensor), \
            "Parameter `process_tensor` is neither a process tensor nor a " \
                + "list of process tensors. "

    if len(process_tensors) > 1:
        assert (initial_state is not None), \
        "For multiple environments an initial state must be specified."

    hs_dim = system.dimension
    for pt in process_tensors:
        assert hs_dim == pt.hilbert_space_dimension

    lens = [len(pt) for pt in process_tensors]
    assert len(set(lens)) == 1, \
        "All process tensors must be of the same length."

    if control is not None:
        assert isinstance(control, Control), \
            "Parameter `control` is not of type `oqupy.Control`."
        tmp_control = control
    else:
        tmp_control = Control(hs_dim)

    dts = [pt.dt for pt in process_tensors]
    assert len(set(dts)) == 1, \
        "All process tensors must be calculated with same timestep."
    dt = dts[0]
    if dt is None:
        raise ValueError("Process tensor has no timestep, "\
            + "please specify time step 'dt'.")

    try:
        tmp_start_time = float(start_time)
    except Exception as e:
        raise AssertionError("Start time must be a float.") from e

    if initial_state is not None:
        assert initial_state.shape == (hs_dim, hs_dim)

    if num_steps is not None:
        try:
            tmp_num_steps = int(num_steps)
        except Exception as e:
            raise AssertionError("Number of steps must be an integer.") from e
    else:
        tmp_num_steps = None

    # -- compute dynamics --

    def propagators(step: int):
        """Create the system propagators (first and second half) for the
        time step `step`. """
        t = tmp_start_time + step * dt
        first_step = expm(system.liouvillian(t+dt/4.0)*dt/2.0)
        second_step = expm(system.liouvillian(t+dt*3.0/4.0)*dt/2.0)
        return first_step, second_step

    def controls(step: int):
        """Create the system (pre and post measurement) for the time step
        `step`. """
        return tmp_control.get_controls(
            step,
            dt=dt,
            start_time=tmp_start_time)


    states = _compute_dynamics(process_tensors=process_tensors,
                               propagators=propagators,
                               controls=controls,
                               initial_state=initial_state,
                               num_steps=tmp_num_steps,
                               record_all=record_all)
    if record_all:
        times = tmp_start_time + np.arange(len(states))*dt
    else:
        times = [tmp_start_time + len(states)*dt]

    return Dynamics(times=list(times),states=states)


def _build_cap(
        process_tensors: List[BaseProcessTensor],
        step: int):
    """
    Builds caps for multiple process tensors at a given timestep.

    Parameters
    ----------
    process_tensors: List[BaseProcessTensor]
        List of process tensor objects.
    step: int
        Step at which to build caps.

    Returns
    -------
    cap_nodes: List[tensornetwork.Node]
        List of caps for each process tensor at the chosen step.

    """
    cap_nodes = []
    for i in range(len(process_tensors)):
        try:
            cap = process_tensors[i].get_cap_tensor(step)
            cap_node = tn.Node(cap)
            cap_nodes.append(cap_node)
        except Exception as e:
            raise ValueError("There are either no cap tensors in "\
                    +f"process tensor {i} or process tensor {i} is "\
                    +"not long enough") from e
        if cap is None:
            raise ValueError(f"Process tensor {i} has no cap tensor "\
                +f"for step {step}.")
    return cap_nodes

def _build_mpo_node(
        process_tensors: List[BaseProcessTensor],
        step: int):
    """
    Contracts MPO for multiple process tensors at a given timestep into a
    single tensor.

    Parameters
    ----------
    process_tensors: List[BaseProcessTensor]
        List of process tensor objects.
    step: int
        Step at which to build MPO.

    Returns
    -------
    mpo_node: tensornetwork.Node
        Single tensor built from MPO for each process tensor at the chosen
        step.

    """
    try:
        mpo = process_tensors[0].get_mpo_tensor(step)
    except Exception as e:
        raise ValueError("Process tensor 0 is not long enough") from e
    if mpo is None:
        raise ValueError("Process tensor 0 has no mpo tensor "\
            +f"for step {step}.")
    mpo_node = tn.Node(mpo)
    for i in range(1,len(process_tensors)):
        try:
            dummy_mpo = process_tensors[i].get_mpo_tensor(step)
        except Exception as e:
            raise ValueError(f"Process tensor {i} is not long enough")\
                from e
        if dummy_mpo is None:
            raise ValueError(f"Process tensor {i} has no mpo tensor "\
                +f"for step {step}.")

        dummy_mpo_node = tn.Node(dummy_mpo)
        mpo_node[-1] ^ dummy_mpo_node[2]
        mpo_node = mpo_node @ dummy_mpo_node
    return mpo_node

def _compute_dynamics(
        process_tensors: List[BaseProcessTensor],
        propagators: Callable[[int], Tuple[ndarray, ndarray]],
        controls: Callable[[int], Tuple[ndarray, ndarray]],
        initial_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> List[ndarray]:
    """ToDo """
    num_envs = len(process_tensors)
    hs_dim = process_tensors[0].hilbert_space_dimension


    initial_tensor = process_tensors[0].get_initial_tensor()
    assert (initial_state is None) ^ (initial_tensor is None), \
        "Initial state must be either (exclusively) encoded in the " \
        + "process tensor or given as an argument."
    if (initial_tensor is None) or (num_envs > 1):
        initial_tensor = util.add_singleton(
            initial_state.reshape(hs_dim**2), 0)

    dummy_init = util.add_singleton(identity(hs_dim**2), 0)
    current = tn.Node(initial_tensor)

    for i in range(num_envs-1):
        dummy = tn.Node(dummy_init)
        current[-1] ^ dummy[1]
        current = current @ dummy

    current_bond_legs = current[:-1]
    current_state_leg = current[-1]
    states = []
    if num_steps is None:
        tmp_num_steps = len(process_tensors[0])
    else:
        tmp_num_steps = num_steps

    for step in range(tmp_num_steps+1):
        # -- apply pre measurement control --
        pre_measurement_control, post_measurement_control = controls(step)
        if pre_measurement_control is not None:
            pre_node = tn.Node(pre_measurement_control.T)
            current_state_leg ^ pre_node[0]
            current_state_leg = pre_node[1]
            current = current @ pre_node

        if step == tmp_num_steps:
            break

        # -- extract current state --
        if record_all:
            cap_nodes = _build_cap(process_tensors, step)
            node_dict, edge_dict = tn.copy([current])
            for i in range(num_envs):
                edge_dict[current_bond_legs[i]] ^ cap_nodes[i][0]
                node_dict[current] = node_dict[current] @ cap_nodes[i]
            state_node = node_dict[current]
            state = state_node.get_tensor().reshape(hs_dim, hs_dim)
            states.append(state)

        # -- apply post measurement control --
        if post_measurement_control is not None:
            post_node = tn.Node(post_measurement_control.T)
            current_state_leg ^ post_node[0]
            current_state_leg = post_node[1]
            current = current @ post_node

        # -- propagate one time step --
        mpo_node = _build_mpo_node(process_tensors, step)
        mpo_bond_legs = [mpo_node[0]]
        mpo_bond_legs += [mpo_node[i*2+1] for i in range(1, num_envs)]

        first_half_prop, second_half_prop = propagators(step)
        first_half_prop_node = tn.Node(first_half_prop.T)
        second_half_prop_node = tn.Node(second_half_prop.T)

        lams = [process_tensors[i].get_lam_tensor(step) \
                for i in range(num_envs)]

        for i in range(num_envs):
            current_bond_legs[i] ^ mpo_bond_legs[i]
        current_state_leg ^ first_half_prop_node[0]
        first_half_prop_node[1] ^ mpo_node[2]
        mpo_node[-1] ^ second_half_prop_node[0]
        current_bond_legs[0] = mpo_node[1]
        for i in range(1,num_envs):
            current_bond_legs[i] = mpo_node[2*i+2]
        current_state_leg = second_half_prop_node[1]
        current = current \
            @ first_half_prop_node @ mpo_node @ second_half_prop_node
        for i,lam in enumerate(lams):
            if lam is not None:
                lam_node = tn.Node(lam)
                current_bond_legs[i] ^ lam_node[0]
                current_bond_legs[i] = lam_node[1]
                current @ lam_node

    # -- extract last state --
    cap_nodes = _build_cap(process_tensors, tmp_num_steps)
    for i in range(num_envs):
        current_bond_legs[i] ^ cap_nodes[i][0]
        current = current @ cap_nodes[i]
    final_state_node = current
    final_state = final_state_node.get_tensor().reshape(hs_dim, hs_dim)
    states.append(final_state)

    return states


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
    with progress(num_steps) as prog_bar:
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
        num_steps=max_step)
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

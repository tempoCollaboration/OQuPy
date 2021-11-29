# Copyright 2021 The TEMPO Collaboration
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
Module on the discrete time evolution of a density matrix.
"""

from typing import List, Optional, Tuple, Callable, Union

import numpy as np
from numpy import ndarray
import tensornetwork as tn
from scipy.linalg import expm

from oqupy import util
from oqupy.dynamics import Dynamics
from oqupy.process_tensor import BaseProcessTensor
from oqupy.system import BaseSystem
from oqupy.operators import identity


def compute_dynamics(
        system: BaseSystem,
        process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor],
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
        initial_state: Optional[ndarray] = None,
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

    Returns
    -------
    dynamics: Dynamics
        The system dynamics for the given system Hamiltonian
        (accounting for the interaction with the environment).
    """
    # -- input parsing --
    assert isinstance(system, BaseSystem), \
        "Parameter `system` is not of type `tempo.BaseSystem`."

    if not isinstance(process_tensor,List):
        process_tensors = [process_tensor]
    else:
        process_tensors = process_tensor

    if len(process_tensors) > 1:
        assert (initial_state is not None), \
        "For multiple environments an initial state must be specified."

    hs_dim = system.dimension
    for pt in process_tensors:
        assert hs_dim == pt.hilbert_space_dimension

    lens = [len(pt) for pt in process_tensors]
    assert len(set(lens)) == 1, \
        "All process tensors must be of the same length."

    if dt is None:
        dts = [pt.dt for pt in process_tensors]
        assert len(set(dts)) == 1, \
            "All process tensors must be calculated with same timestep."
        dt = dts[0]
        if dt is None:
            raise ValueError("Process tensor has no timestep, "\
                + "please specify time step 'dt'.")
    try:
        __dt = float(dt)
    except Exception as e:
        raise AssertionError("Time step 'dt' must be a float.") from e

    try:
        __start_time = float(start_time)
    except Exception as e:
        raise AssertionError("Start time must be a float.") from e

    if initial_state is not None:
        assert initial_state.shape == (hs_dim, hs_dim)

    if num_steps is not None:
        try:
            __num_steps = int(num_steps)
        except Exception as e:
            raise AssertionError("Number of steps must be an integer.") from e
    else:
        __num_steps = None

    # -- compute dynamics --

    def propagators(step: int):
        """Create the system propagators (first and second half) for the
        time step `step`. """
        t = __start_time + step * __dt
        first_step = expm(system.liouvillian(t+__dt/4.0)*__dt/2.0).T
        second_step = expm(system.liouvillian(t+__dt*3.0/4.0)*__dt/2.0).T
        return first_step, second_step

    states = _compute_dynamics(process_tensors=process_tensors,
                               controls=propagators,
                               initial_state=initial_state,
                               num_steps=__num_steps,
                               record_all=record_all)
    if record_all:
        times = __start_time + np.arange(len(states))*__dt
    else:
        times = [__start_time + len(states)*__dt]

    return Dynamics(times=list(times),states=states)


def compute_final_state(
        system: BaseSystem,
        process_tensor: Union[List[BaseProcessTensor],BaseProcessTensor],
        start_time: Optional[float] = 0.0,
        dt: Optional[float] = None,
        initial_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None) -> ndarray:
    """
    ToDo.
    """
    dynamics = compute_dynamics(
        system=system,
        process_tensor=process_tensor,
        start_time=start_time,
        dt=dt,
        initial_state=initial_state,
        num_steps=num_steps,
        record_all=False)
    return dynamics.states[-1]


def _compute_dynamics(
        process_tensors: List[BaseProcessTensor],
        controls: Callable[[int], Tuple[ndarray, ndarray]],
        initial_state: Optional[ndarray] = None,
        num_steps: Optional[int] = None,
        record_all: Optional[bool] = True) -> List[ndarray]:
    """See BaseProcessTensorBackend.compute_dynamics() for docstring. """
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

    current_bond_legs = [current[i] for i in range(num_envs)]
    current_state_leg = current[-1]
    states = []
    if num_steps is None:
        __num_steps = len(process_tensors[0])
    else:
        __num_steps = num_steps

    for step in range(__num_steps):
        if record_all:
            # -- extract current state --
            cap_nodes = []
            for i in range(num_envs):
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
            node_dict, edge_dict = tn.copy([current])
            for i in range(num_envs):
                edge_dict[current_bond_legs[i]] ^ cap_nodes[i][0]
                node_dict[current] = node_dict[current] @ cap_nodes[i]
            state_node = node_dict[current]
            state = state_node.get_tensor().reshape(hs_dim, hs_dim)
            states.append(state)

        # -- propagate one time step --
        try:
            mpo = process_tensors[0].get_mpo_tensor(step)
        except Exception as e:
            raise ValueError("Process tensor 0 is not long enough") from e
        if mpo is None:
            raise ValueError("Process tensor 0 has no mpo tensor "\
                +f"for step {step}.")
        mpo_node = tn.Node(mpo)
        mpo_bond_legs = [mpo_node[0]]
        for i in range(1,num_envs):
            try:
                dummy_mpo = process_tensors[i].get_mpo_tensor(step)
            except Exception as e:
                raise ValueError(f"Process tensor {i} is not long enough")\
                    from e
            if dummy_mpo is None:
                raise ValueError(f"Process tensor {i} has no mpo tensor "\
                    +f"for step {step}.")

            dummy_mpo_node = tn.Node(process_tensors[i].get_mpo_tensor(step))
            mpo_node[-1] ^ dummy_mpo_node[2]
            mpo_node = mpo_node @ dummy_mpo_node
            mpo_bond_legs.append(mpo_node[i*2+1])

        pre, post = controls(step)
        pre_node = tn.Node(pre)
        post_node = tn.Node(post)

        lams = [process_tensors[i].get_lam_tensor(step) for i in range(num_envs)]

        for i in range(num_envs):
            current_bond_legs[i] ^ mpo_bond_legs[i]
        current_state_leg ^ pre_node[0]
        pre_node[1] ^ mpo_node[2]
        mpo_node[-1] ^ post_node[0]
        current_bond_legs[0] = mpo_node[1]
        for i in range(1,num_envs):
            current_bond_legs[i] = mpo_node[2*i+2]
        current_state_leg = post_node[1]
        current = current @ pre_node @ mpo_node @ post_node
        for i,lam in enumerate(lams):
            if lam is not None:
                lam_node = tn.Node(lam)
                current_bond_legs[i] ^ lam_node[0]
                current_bond_legs[i] = lam_node[1]
                current @ lam_node

    # -- extract last state --
    cap_nodes = []
    for i in range(num_envs):
        cap = process_tensors[i].get_cap_tensor(__num_steps)
        if cap is None:
            raise ValueError(f"Process tensor {i} has no cap tensor "\
                +f"for step {step}.")
        cap_node = tn.Node(cap)
        cap_nodes.append(cap_node)

    for i in range(num_envs):
        current_bond_legs[i] ^ cap_nodes[i][0]
        current = current @ cap_nodes[i]
    final_state_node = current
    final_state = final_state_node.get_tensor().reshape(hs_dim, hs_dim)
    states.append(final_state)

    return states

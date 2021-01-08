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
Module for the process tensor (PT) object. This code is based on [Pollock2018].

**[Pollock2018]**
F.  A.  Pollock,  C.  Rodriguez-Rosario,  T.  Frauenheim,
M. Paternostro, and K. Modi, *Non-Markovian quantumprocesses: Complete
framework and efficient characterization*, Phys. Rev. A 97, 012127 (2018).
"""

from typing import Dict, List, Optional, Text, Union

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

from time_evolving_mpo.backends.backend_factory import \
    get_process_tensor_backend
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NpDtype, NpDtypeReal
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.file_formats import assert_process_tensor_dict
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.util import save_object, load_object


class ProcessTensor(BaseAPIClass):
    """
    Represents a specific process tensor.

    If the field `times` is `None` this amounts to no information on
    the time slots of the process tensor (only "initial step", "first step",
    etc). If the field `times` is a `float` it signals that the time steps are
    uniformly spaced. If the field `times` is an `numpy.ndarray` it
    has to be a vector of the time slots considered in this process tensor in
    ascending order. In this case the length of `times` must be the length of
    `tensors` plus 1.

    If the field `initial_tensor` is `None` this amounts to no given initial
    state. If the field `initial_tensor` is an `numpy.ndarray` it must be a
    2-legged tensor (i.e. a matrix) where the first leg is the internal leg
    connecting to the next part of the array of tensors that represent the
    process tensor. The second leg is vectorised initial state (in fact the
    first slot in the process tensor).

    The field `tensors` is list of three or four legged tensors. The first and
    second legs are the internal legs that connect to the previous and next
    tensor. If `initial_tensor` is `None` the first leg of the first tensor
    must be a dummy leg of dimension 1. The  second leg of the last tensor must
    always be a dummy leg of dimension 1. The third leg is the "incoming" leg
    of the previous time slot, while the fourth leg is the "resulting" leg of
    the following time slot. If the tensor has only three legs, a
    Kronecker-delta between the third and fourth leg is assumed.


    Parameters
    ----------
    times: ndarray / float / None
        Time slots of process tensor. See description above.
    tensors: list(ndarray)
        Process tensor tensors in MPS form. See description above.
    initial_tensor: ndarray
        Initial tensor of process tensor in MPS form. See description above.
    backend: str (default = None)
        The name of the backend to use foe computations. If
        `backend` is ``None`` then the default backend is used.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str
        An optional name for the process tensor.
    description: str
        An optional description of the process tensor.
    description_dict: dict
        An optional dictionary with descriptive data.

    """
    def __init__(
            self,
            times: Union[float, ndarray],
            tensors: List[ndarray],
            initial_tensor: Optional[ndarray] = None,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a ProcessTensor object. """

        self._backend_class, self._backend_config = \
            get_process_tensor_backend(backend, backend_config)

        p_t_dict =  {
            "version":"1.0",
            "name":name,
            "description":description,
            "description_dict":description_dict,
            "times":times,
            "initial_tensor":initial_tensor,
            "tensors":tensors,
            }
        assert_process_tensor_dict(p_t_dict)

        self._times = times

        if isinstance(times, ndarray):
            self._times_array = times.astype(NpDtypeReal)
        elif isinstance(times, float):
            assert times > 0.0
            self._times_array =  \
                    times * np.arange(len(tensors)+1).astype(NpDtypeReal)
        elif times is None:
            self._times_array = np.arange(len(tensors)+1).astype(NpDtypeReal)
        # else:
        #     raise AssertionError("Parameter `times` must be `None` or " \
        #         + "of type `float` or `ndarray`")

        if len(tensors)>0:
            dim = int(np.sqrt(tensors[0].shape[2]))
            trace = np.identity(dim)/float(dim)
            trace = np.sqrt(trace.reshape(dim**2))
        else:
            trace = np.array([0])

        self._backend_instance = self._backend_class(
                tensors=tensors,
                initial_tensor=initial_tensor,
                trace=trace,
                config=self._backend_config)

        super().__init__(name, description, description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  length        = {} timesteps \n".format(len(self)))
        if len(self) > 0:
            ret.append("  min time      = {} \n".format(
                np.min(self._times)))
            ret.append("  max time      = {} \n".format(
                np.max(self._times)))
        return "".join(ret)

    def __len__(self) -> int:
        return len(self._times)

    @property
    def times(self) -> ndarray:
        """Times of the dynamics. """
        return self._times_array.copy()

    def get_bond_dimensions(self) -> ndarray:
        """Return the bond dimensions of the MPS form of the process tensor."""
        return self._backend_instance.get_bond_dimensions()

    def export(
            self,
            filename: Text,
            overwrite: bool = False) -> None:
        """
        Save process tensor to a file (format ProcessTensorFormat version 1.0).

        Parameters
        ----------
        filename: str
            Path and filename to file that should be created.
        overwrite: bool (default = False)
            If set `True` then file is overwritten in case it already exists.
        """
        tensors, initial_tensor = self._backend_instance.export_tensors()
        p_t_dict = {
            "version": "1.0",
            "name": self.name,
            "description": self.description,
            "description_dict": self.description_dict,
            "times": self.times,
            "initial_tensor":initial_tensor,
            "tensors":tensors,
            }
        assert_process_tensor_dict(p_t_dict)
        save_object(p_t_dict, filename, overwrite)

    def compute_dynamics_from_system(
            self,
            system: BaseSystem,
            initial_state: Optional[ndarray] = None) -> Dynamics:
        """
        Compute the system dynamics for a given system Hamiltonian.

        Parameters
        ----------
        system: BaseSystem
            Object containing the system Hamiltonian information.

        Returns
        -------
        dynamics: Dynamics
            The system dynamics for the given system Hamiltonian
            (accounting for the interaction with the environment).
        """
        assert isinstance(system, BaseSystem), \
            "Parameter `system` is not of type `tempo.BaseSystem`."

        dimension = system.dimension

        if initial_state is None:
            initial_state_vector = None
        else:
            try:
                _initial_state = np.array(initial_state, dtype=NpDtype)
                _initial_state.setflags(write=False)
            except Exception as e:
                raise AssertionError("Initial state must be numpy array.") \
                    from e
            assert len(_initial_state.shape) == 2, \
                "Initial state is not a matrix."
            assert _initial_state.shape[0] == \
                _initial_state.shape[1], \
                "Initial state is not a square matrix."

            initial_state_vector = _initial_state.reshape(dimension**2)

        def propagators(step: int):
            """Create the system propagators (first and second half) for the
            time step `step`. """
            dt = self._times_array[step+1] - self._times_array[step]
            t = self._times_array[step]
            first_step = expm(system.liouvillian(t+dt/4.0)*dt/2.0).T
            second_step = expm(system.liouvillian(t+dt*3.0/4.0)*dt/2.0).T
            return first_step, second_step

        state_vectors = self._backend_instance.compute_dynamics(
                            controls=propagators,
                            initial_state=initial_state_vector)

        states = [v.reshape((dimension,dimension)) for v in state_vectors]
        times = list(self._times_array)

        dyn = Dynamics(times=times, states=states)
        return dyn

    def compute_final_state_from_system(
            self,
            system: BaseSystem,
            initial_state: Optional[ndarray] = None) -> Dynamics:
        """
        Compute final state for a given system Hamiltonian.

        Parameters
        ----------
        system: BaseSystem
            Object containing the system Hamiltonian information.

        Returns
        -------
        final_state: ndarray
            The final system system state for the given system Hamiltonian
            (accounting for the interaction with the environment).
        """

        assert isinstance(system, BaseSystem), \
            "Parameter `system` is not of type `tempo.BaseSystem`."

        dimension = system.dimension

        if initial_state is None:
            initial_state_vector = None
        else:
            try:
                _initial_state = np.array(initial_state, dtype=NpDtype)
                _initial_state.setflags(write=False)
            except Exception as e:
                raise AssertionError("Initial state must be numpy array.") \
                    from e
            assert len(_initial_state.shape) == 2, \
                "Initial state is not a matrix."
            assert _initial_state.shape[0] == \
                _initial_state.shape[1], \
                "Initial state is not a square matrix."

            initial_state_vector = _initial_state.reshape(dimension**2)

        def propagators(step: int):
            """Create the system propagators (first and second half) for the
            time step `step`. """
            dt = self._times_array[step+1] - self._times_array[step]
            t = self._times_array[step]
            first_step = expm(system.liouvillian(t+dt/4.0)*dt/2.0).T
            second_step = expm(system.liouvillian(t+dt*3.0/4.0)*dt/2.0).T
            return first_step, second_step

        final_state_vector = self._backend_instance.compute_final_state(
                            controls=propagators,
                            initial_state=initial_state_vector)

        final_state = final_state_vector.reshape(dimension, dimension)

        return final_state


def import_process_tensor(filename: Text) -> ProcessTensor:
    """
    Load process tensor from a file (format ProcessTensorFormat version 1.0).

    Parameters
    ----------
    filename: str
        Path and filename to file that should read in.

    Returns
    -------
    process_tensor: ProcessTensor
        The process tensor stored in the file `filename`.
    """
    p_t = load_object(filename)
    assert "version" in p_t, \
        "Can't import process tensor from file {} ".format(filename) \
        + "because it doesn't have a 'version' field."
    assert p_t["version"] == "1.0", \
        "Can't import process tensor from file {} ".format(filename) \
        + "as it appears to be an incompatible version."
    assert_process_tensor_dict(p_t)

    return ProcessTensor(times=p_t["times"],
                         tensors=list(p_t["tensors"]),
                         initial_tensor=p_t["initial_tensor"],
                         name=p_t["name"],
                         description=p_t["description"],
                         description_dict=p_t["description_dict"])

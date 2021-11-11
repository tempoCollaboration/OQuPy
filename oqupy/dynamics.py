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

from typing import Dict, List, Optional, Text, Tuple
from copy import copy

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype, NpDtypeReal
from oqupy.file_formats import assert_tempo_dynamics_dict
from oqupy.util import save_object, load_object


class Dynamics(BaseAPIClass):
    """
    Represents a specific time evolution of a density matrix.

    Parameters
    ----------
    times: List[float] (default = None)
        A list of points in time.
    states: List[ndarray] (default = None)
        A list of states at the times `times`.
    name: str
        An optional name for the dynamics.
    description: str
        An optional description of the dynamics.
    description_dict: dict
        An optional dictionary with descriptive data.
    """
    def __init__(
            self,
            times: Optional[List[float]] = None,
            states: Optional[List[ndarray]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a Dynamics object. """
        # input check times and states
        if times is None:
            times = []
        if states is None:
            states = []
        assert isinstance(times, list), \
            "Argument `times` must be a list."
        assert isinstance(states, list), \
            "Argument `states` must be a list."
        assert len(times) == len(states), \
            "Lists `times` and `states` must have the same length."
        self._times = []
        self._states = []
        self._expectation_operators = []
        self._expectation_lists = []
        self._shape = None
        for time, state in zip(times, states):
            self.add(time, state)

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

    def _sort(self) -> None:
        """Sort the time evolution (chronologically). """
        tuples = zip(self._times, self._states)
        __times, __states = zip(*sorted(tuples)) # ToDo: make more elegant
        self._times = list(__times)
        self._states = list(__states)

    @property
    def times(self) -> ndarray:
        """Times of the dynamics. """
        return np.array(self._times, dtype=NpDtypeReal)

    @property
    def states(self) -> ndarray:
        """States of the dynamics. """
        return np.array(self._states, dtype=NpDtype)

    @property
    def shape(self) -> ndarray:
        """Numpy shape of the states. """
        return copy(self._shape)

    def add(
            self,
            time: float,
            state: ndarray) -> None:
        """
        Append a state at a specific time to the time evolution.

        Parameters
        ----------
        time: float
            The point in time.
        state: ndarray
            The state at the time `time`.
        """
        try:
            __time = float(time)
        except Exception as e:
            raise AssertionError("Argument `time` must be float.") from e
        try:
            __state = np.array(state, dtype=NpDtype)
        except Exception as e:
            raise AssertionError("Argument `state` must be ndarray.") from e
        if self._shape is None:
            __shape = __state.shape
            assert len(__shape) == 2, \
                "State must be a square matrix. " \
                + "But the dimensions are {}.".format(__shape)
            assert __shape[0] == __shape[1], \
                "State must be a square matrix. " \
                + "But the dimensions are {}.".format(__shape)
            self._shape = __shape
        else:
            assert __state.shape == self._shape, \
                "Appended state doesn't have the same shape as previous " \
                + "states ({}, but should be {})".format(__state.shape,
                                                         self._shape)

        self._times.append(__time)
        self._states.append(__state)

        # ToDo: do this more elegantly and less resource draining.
        if len(self) > 1 and (self._times[-1] < np.max(self._times[:-1])):
            self._sort()

    def export(
            self,
            filename: Text,
            overwrite: bool = False) -> None:
        """
        Save dynamics to a file (format TempoDynamicsFormat version 1.0).

        Parameters
        ----------
        filename: str
            Path and filename to file that should be created.
        overwrite: bool (default = False)
            If set `True` then file is overwritten in case it already exists.
        """
        dyn = {"version": "1.0",
               "name": self.name,
               "description": self.description,
               "description_dict": self.description_dict,
               "times": self.times,
               "states": self.states}
        assert_tempo_dynamics_dict(dyn)
        save_object(dyn, filename, overwrite)

    def expectations(
            self,
            operator: Optional[ndarray] = None,
            real: Optional[bool] = False) -> Tuple[ndarray, ndarray]:
        r"""
        Return the time evolution of the expectation value of specific
        operator. The expectation for :math:`t` is

        .. math::

            \langle \hat{O}(t) \rangle = \mathrm{Tr}\{ \hat{O} \rho(t) \}

        with `operator` :math:`\hat{O}`.

        Parameters
        ----------
        operator: ndarray (default = None)
            The operator :math:`\hat{O}`. If `operator` is `None` then the
            trace of :math:`\rho(t)` is returned.
        real: bool (default = False)
            If set True then only the real part of the expectation is returned.

        Returns
        -------
        times: ndarray
            The points in time :math:`t`.
        expectations: ndarray
            Expectation values :math:`\langle \hat{O}(t) \rangle`.
        """
        if len(self) == 0:
            return None, None
        if operator is None:
            __operator = np.identity(self._shape[0], dtype=NpDtype)
        else:
            try:
                __operator = np.array(operator, dtype=NpDtype)
            except Exception as e:
                raise AssertionError("Argument `operator` must be ndarray.") \
                    from e
            assert __operator.shape == self._shape, \
                "Argument `operator` must have the same shape as the " \
                + "states. Has shape {}, ".format(__operator.shape) \
                + "but should be {}.".format(self._shape)

        operator_index = next((i for i, op in \
            enumerate(self._expectation_operators) if \
            np.array_equal(op, __operator)), -1)
        if operator_index == -1: # Operator not seen before
            self._expectation_operators.append(__operator)
            self._expectation_lists.append([])

        expectations_list = self._expectation_lists[operator_index]

        for state in self._states[len(expectations_list):]:
            expectations_list.append(np.trace(__operator @ state))

        self._expectation_lists[operator_index] = expectations_list

        times = np.array(self._times)
        if real:
            expectations = np.real(np.array(expectations_list))
        else:
            expectations = np.array(expectations_list)
        return times, expectations

# def distance(*args, **kwargs): # ToDo
#     """
#     ToDo
#     """
#     pass # ToDo
#     return NotImplemented, NotImplemented #ToDo


def import_dynamics(filename: Text) -> Dynamics:
    """
    Load dynamics from a file (format TempoDynamicsFormat version 1.0).

    Parameters
    ----------
    filename: str
        Path and filename to file that should read in.

    Returns
    -------
    dynamics: Dynamics
        The time evolution stored in the file `filename`.
    """
    dyn = load_object(filename)
    assert "version" in dyn, \
        "Can't import dynamics from file {} ".format(filename) \
        + "because it doesn't have a 'version' field."
    assert dyn["version"] == "1.0", \
        "Can't import dynamics from file {} ".format(filename) \
        + "as it appears to be an incompatible version."
    assert_tempo_dynamics_dict(dyn)
    return Dynamics(times=list(dyn["times"]),
                    states=list(dyn["states"]),
                    name=dyn["name"],
                    description=dyn["description"],
                    description_dict=dyn["description_dict"])


# def norms(*args, **kwargs): # ToDo
#     """
#     ToDo
#     """
#     pass # ToDo
#     return NotImplemented #ToDo

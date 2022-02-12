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
Module on the discrete time evolution of a density matrix.
"""

from typing import List, Optional, Text, Tuple
from copy import copy
from bisect import bisect

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype, NpDtypeReal

class BaseDynamics(BaseAPIClass):
    """
    Base class for objects recording the dynamics of an open quantum
    system. Consists at least of a list of times at which the dynamics
    has been computed and a list of states describing the system density
    matrix at those times.

    Parameters
    ----------
    name: str
        An optional name for the dynamics.
    description: str
        An optional description of the dynamics.
    """
    def __init__(
            self,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        super().__init__(name, description)
        self._times = []
        self._states = []
        self._expectation_operators = []
        self._expectation_lists = []
        self._shape = None

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

class Dynamics(BaseDynamics):
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
    """
    def __init__(
            self,
            times: Optional[List[float]] = None,
            states: Optional[List[ndarray]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a Dynamics object. """
        super().__init__(name, description)
        times, states = _parse_times_states(times, states)
        for time, state in zip(times, states):
            self.add(time, state)

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
        tmp_time = _parse_time(time)
        tmp_state, tmp_shape = _parse_state(state, self._shape)
        index = _find_list_index(self._times, tmp_time)
        self._times.insert(index, tmp_time)
        self._states.insert(index, tmp_state)
        if self._shape is None:
            self._shape = tmp_shape


class DynamicsWithField(BaseDynamics):
    """
    Represents a specific time evolution of a density matrix together
    with a coherent field.

    Parameters
    ----------
    times: List[float] (default = None)
        A list of points in time.
    states: List[ndarray] (default = None)
        A list of states at the times `times`.
    fields: List[complex] (default = None)
        A list of fields at the times `times`.
    name: str
        An optional name for the dynamics.
    description: str
        An optional description of the dynamics.
    """
    def __init__(
            self,
            times: Optional[List[float]] = None,
            states: Optional[List[ndarray]] = None,
            fields: Optional[List[complex]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a DynamicsWithField object"""
        super().__init__(name, description)
        self._fields = []
        times, states = _parse_times_states(times, states)
        times, fields = _parse_times_fields(times, fields)
        for time, state, field in zip(times, states, fields):
            self.add(time, state, field)

    @property
    def fields(self) -> ndarray:
        """Fields of the dynamics. """
        return np.array(self._fields, dtype=NpDtype)

    def field_expectations(self) -> Tuple[ndarray, ndarray]:
        r"""
        Return the time evolution of the coherent field.

        Returns
        -------
        times: ndarray
            The points in time :math:`t`.
        field_expectations: ndarray
            Values :math:`\langle a(t) \rangle`.
        """
        if len(self) == 0:
            return None, None
        return np.array(copy(self._times), dtype=NpDtypeReal), np.array(copy(self._fields),
                dtype=NpDtype)

    def add(
            self,
            time: float,
            state: ndarray,
            field: complex) -> None:
        """
        Append a state and field at a specific time to the time evolution.

        Parameters
        ----------
        time: float
            The point in time.
        state: ndarray
            The state at the time `time`.
        field: complex
            The field at the time `time`.
        """
        tmp_time = _parse_time(time)
        tmp_state, tmp_shape = _parse_state(state, self._shape)
        tmp_field = _parse_field(field)
        index = _find_list_index(self._times, tmp_time)
        self._times.insert(index, tmp_time)
        self._states.insert(index, tmp_state)
        self._fields.insert(index, tmp_field)
        if self._shape is None:
            self._shape = tmp_shape

def _parse_times_states(times, states) -> Tuple[List[float],
        List[ndarray]] :
    """Check times and states are None or lists of the same length"""
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
    return times, states

def _parse_times_fields(times, fields) -> Tuple[List[float],
        List[complex]]:
    """Check times and fields are None or lists of the same length"""
    if times is None:
        times = []
    if fields is None:
        fields = []
    assert isinstance(times, list), \
        "Argument `times` must be a list."
    assert isinstance(fields, list), \
        "Argument `fields` must be a list."
    assert len(times) == len(fields), \
        "Lists `times` and `fields` must have the same length."
    return times, fields

def _parse_time(time) -> float:
    try:
        tmp_time = float(time)
    except Exception as e:
        raise AssertionError("Argument `time` must be float.") from e
    return tmp_time

def _parse_state(state, previous_shape) -> Tuple[ndarray, Tuple[int]]:  
    try:
        tmp_state = np.array(state, dtype=NpDtype)
    except Exception as e:
        raise AssertionError("Argument `state` must be ndarray.") from e
    tmp_shape = tmp_state.shape
    if previous_shape is not None:
        assert tmp_state.shape == previous_shape, \
            "Appended state doesn't have the same shape as previous " \
            + "states ({}, but should be {})".format(tmp_state.shape,
                                                         previous_shape)
    assert len(tmp_shape) == 2, \
            "State must be a square matrix. " \
            + "But the dimensions are {}.".format(tmp_shape)
    assert tmp_shape[0] == tmp_shape[1], \
            "State must be a square matrix. " \
            + "But the dimensions are {}.".format(tmp_shape)
    return tmp_state, tmp_shape

def _parse_field(field) -> complex:
    try:
        tmp_field = complex(field)
    except Exception as e:
        raise AssertionError("Argument `field` must be complex.") from e
    return tmp_field

def _find_list_index(sorted_list, entry_value) -> int:
    """
    Return `index` such that a `sorted_list` stays sorted when extended with
    `entry_value` like so:
       `sorted_list.insert(index, entry_value)`
    """
    return bisect(sorted_list, entry_value)

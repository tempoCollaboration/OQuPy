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

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype, NpDtypeReal


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
    """
    def __init__(
            self,
            times: Optional[List[float]] = None,
            states: Optional[List[ndarray]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
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

        super().__init__(name, description)

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
        tmp_times, tmp_states = zip(*sorted(tuples)) # ToDo: make more elegant
        self._times = list(tmp_times)
        self._states = list(tmp_states)

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
            tmp_time = float(time)
        except Exception as e:
            raise AssertionError("Argument `time` must be float.") from e
        try:
            tmp_state = np.array(state, dtype=NpDtype)
        except Exception as e:
            raise AssertionError("Argument `state` must be ndarray.") from e
        if self._shape is None:
            tmp_shape = tmp_state.shape
            assert len(tmp_shape) == 2, \
                "State must be a square matrix. " \
                + "But the dimensions are {}.".format(tmp_shape)
            assert tmp_shape[0] == tmp_shape[1], \
                "State must be a square matrix. " \
                + "But the dimensions are {}.".format(tmp_shape)
            self._shape = tmp_shape
        else:
            assert tmp_state.shape == self._shape, \
                "Appended state doesn't have the same shape as previous " \
                + "states ({}, but should be {})".format(tmp_state.shape,
                                                         self._shape)

        self._times.append(tmp_time)
        self._states.append(tmp_state)

        # ToDo: do this more elegantly and less resource draining.
        if len(self) > 1 and (self._times[-1] < np.max(self._times[:-1])):
            self._sort()

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
            tmp_operator = np.identity(self._shape[0], dtype=NpDtype)
        else:
            try:
                tmp_operator = np.array(operator, dtype=NpDtype)
            except Exception as e:
                raise AssertionError("Argument `operator` must be ndarray.") \
                    from e
            assert tmp_operator.shape == self._shape, \
                "Argument `operator` must have the same shape as the " \
                + "states. Has shape {}, ".format(tmp_operator.shape) \
                + "but should be {}.".format(self._shape)

        operator_index = next((i for i, op in \
            enumerate(self._expectation_operators) if \
            np.array_equal(op, tmp_operator)), -1)
        if operator_index == -1: # Operator not seen before
            self._expectation_operators.append(tmp_operator)
            self._expectation_lists.append([])

        expectations_list = self._expectation_lists[operator_index]

        for state in self._states[len(expectations_list):]:
            expectations_list.append(np.trace(tmp_operator @ state))

        self._expectation_lists[operator_index] = expectations_list

        times = np.array(self._times)
        if real:
            expectations = np.real(np.array(expectations_list))
        else:
            expectations = np.array(expectations_list)
        return times, expectations

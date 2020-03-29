# Copyright 2020 The TEMPO Collaboration
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
Module for example backend.
"""

from typing import Callable, Dict, Tuple
from copy import copy
from time import sleep
from warnings import warn

from numpy import ndarray

from time_evolving_mpo.backends.base_backend import BaseBackend
from time_evolving_mpo.backends.base_backend import BaseTempoBackend


class ExampleTempoBackend(BaseTempoBackend):
    """See BaseTempoBackend for docstring. """
    def __init__(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float,
            sleep_time: float):
        """Create a ExampleTempoBackend object. """
        self._initial_state = initial_state
        self._influence = influence
        self._propagators = propagators
        self._sum_north = sum_north
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._step = None
        self._state = None
        self._sleep_time = sleep_time

    def initialize(self) -> Tuple[int, ndarray]:
        """See BaseBackend.initialize() for docstring."""
        self._state = self._initial_state
        self._step = 0
        warn("ExampleTempoBackend only propagates with the system "
             + "Liouvillian and doesn't include the bath.")
        return self._step, copy(self._state)

    def compute_step(self) -> Tuple[int, ndarray]:
        """See BaseBackend.compute_step() for docstring."""

        # -- That's where the actual computation would be -------------------
        # This does the system propagation without the influence of the bath.
        prop_1, prop_2 = self._propagators(self._step)
        self._state = prop_2 @ prop_1 @ self._state
        self._step = self._step + 1
        sleep(self._sleep_time)
        # -------------------------------------------------------------------

        return self._step, copy(self._state)


class ExampleBackend(BaseBackend):
    """See BaseBackend for docstring. """
    def __init__(self, config: Dict) -> None:
        """Create ExampleBackend object. """
        if config is None:
            self._sleep_time = 0.00
        else:
            self._sleep_time = config["sleep_time"]
        self._tempo_backend_class = ExampleTempoBackend

    def get_tempo_backend(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float) -> BaseTempoBackend:
        """Returns an ExampleTempoBackend object. """
        return self._tempo_backend_class(initial_state,
                                         influence,
                                         propagators,
                                         sum_north,
                                         sum_west,
                                         dkmax,
                                         epsrel,
                                         self._sleep_time)

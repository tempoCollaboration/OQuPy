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
Module for base classes of backends.
"""

from typing import Callable, Dict, Optional, Tuple
from numpy import ndarray


class BaseTempoBackend:
    """
    Base class for tempo backends.

    Parameters
    ----------
    initial_state: ndarray
        The initial density matrix (as a vector).
    influence: callable(int) -> ndarray
        Callable that takes an integer `step` and returns the influence super
        operator of that `step`.
    unitary_transform: ndarray
        ToDo
    propagators: callable(int) -> ndarray, ndarray
        Callable that takes an integer `step` and returns the first and second
        half of the system propagator of that `step`.
    sum_north: ndarray
        The summing vector for the north leggs.
    sum_west: ndarray
        The summing vector for the west leggs.
    dkmax: int
        Number of influences to include. If ``dkmax == -1`` then all influences
        are included.
    epsrel: float
        Maximal relative SVD truncation error.
    """
    def __init__(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            unitary_transform: ndarray,
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float):
        """Create a BaseBackend object. """
        self._initial_state = initial_state
        self._influence = influence
        self._unitary_transform = unitary_transform
        self._propagators = propagators
        self._sum_north = sum_north
        self._sum_west = sum_west
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._step = None
        self._state = None

    @property
    def step(self) -> int:
        """The current step in the TEMPO computation. """
        return self._step

    def initialize(self) -> Tuple[int, ndarray]:
        """
        Initializes the TEMPO tensor network.

        Returns
        -------
        step: int = 0
            The current step count, which after initialization, is 0 .
        state: ndarray
            Density matrix (as a vector) at the current step, which after
            initialization, is just the initial state.
        """
        raise NotImplementedError(
            "Class {} has no initialize implementation.".format(
                type(self).__name__))

    def compute_step(self) -> Tuple[int, ndarray]:
        """
        Initializes the TEMPO tensor network.

        Returns
        -------
        step: int
            The current step count.
        state: ndarray
            Density matrix at the current step.
        """
        raise NotImplementedError(
            "Class {} has no compute_step implementation.".format(
                type(self).__name__))


class BaseBackend:
    """
    Base class for backends.

    Parameters
    ----------
    config: dict (default = None)
        Dictionary with information to configure the specific backend.
    """
    def __init__(self, config: Optional[Dict] = None):
        """Create a BaseBackend object."""
        self._tempo_backend_class = BaseTempoBackend
        raise NotImplementedError(
            "Class {} should not call BaseBackend constructor.".format(
                type(self).__name__))

    def get_tempo_backend(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            unitary_transform: ndarray,
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float) -> BaseTempoBackend:
        """Returns an TempoBaseBackend object. """
        return self._tempo_backend_class(initial_state,
                                         influence,
                                         unitary_transform,
                                         propagators,
                                         sum_north,
                                         sum_west,
                                         dkmax,
                                         epsrel)

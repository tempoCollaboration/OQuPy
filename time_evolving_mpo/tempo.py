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
Module on for the original time evolving matrix product operator (TEMPO)
algorithm. This code is based on [Strathearn2018].

**[Strathearn2018]** Strathearn, A., Kirton, P., Kilda, D., Keeling, J.
and Lovett, B.W., 2018. *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators.* Nature communications, 9(1), pp.1-9.
"""

from typing import Dict, Optional, Text
from typing import Any as ArrayLike

from copy import copy
from numpy import array, ndarray

from time_evolving_mpo.bath import Bath
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NP_DTYPE
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.backends.backend_factory import get_backend


class TempoParameters(BaseAPIClass):
    """
    ToDo
    """
    def __init__(self, *args, **kwargs): # ToDo
        """
        ToDo
        """
        pass # ToDo


class Tempo(BaseAPIClass):
    """
    ToDo
    """
    def __init__(
            self,
            system: BaseSystem,
            bath: Bath,
            parameters: TempoParameters,
            initial_state: ArrayLike,
            start_time: float,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """
        ToDo
        """
        self._backend = get_backend(backend, backend_config)

        assert isinstance(system, BaseSystem), \
            "Argument 'system' must be an instance of BaseSystem."
        self._system = system

        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        try:
            __initial_state = array(initial_state, dtype=NP_DTYPE)
            __initial_state.setflags(write=False)
        except:
            raise AssertionError("Initial state must be numpy array")
        assert len(__initial_state.shape) == 2, \
            "Initial state is not a matrix."
        assert __initial_state.shape[0] == \
            __initial_state.shape[1], \
            "Initial state is not a square matrix."
        self._initial_state = __initial_state
        self._dimension = self._initial_state.shape[0]

        assert self._bath.dimension == self._dimension and \
            self._system.dimension == self._dimension, \
            "Hilbertspace dimensions are unequal: " \
            + "system ({}), ".format(self._system.dimension) \
            + "initial state ({}), ".format(self._dimension) \
            + "and bath coupling ({}), ".format(self._bath.dimension)

        super().__init__(name, description, description_dict)

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension. """
        return copy(self._dimension)

    def compute(self, end_time: float) -> None:
        """
        ToDo
        """
        pass # ToDo

    def get_dynamics(self) -> Dynamics:
        """
        ToDo
        """
        pass # ToDo
        return Dynamics() # ToDo


def guess_tempo_parameters(*args, **kwargs): # ToDo
    """
    ToDo
    """
    pass # ToDo
    return TempoParameters() #ToDo


def tempo_compute(
        system: BaseSystem,
        bath: Bath,
        initial_state: ArrayLike,
        start_time: float,
        end_time: float,
        parameters: Optional[TempoParameters] = None,
        backend: Optional[Text] = None,
        backend_config: Optional[Dict] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
        description_dict: Optional[Dict] = None) -> Dynamics:
    """
    ToDo
    """
    pass # ToDo
    return Dynamics() # ToDo

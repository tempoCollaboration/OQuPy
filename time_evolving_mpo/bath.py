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
Module on physical information on the bath and it's coupling to the system.
"""
from typing import Dict, Optional, Text
from typing import Any as ArrayLike
from copy import copy

from numpy import allclose, array, diag, ndarray

from time_evolving_mpo.spectral_density import BaseSD
from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.base_api import BaseAPIClass


class Bath(BaseAPIClass):
    """
    Represents the bath degees of freedom with a specific coupling operator
    (to the system), a specific spectral density and a specific temperature.

    Parameters
    ----------
    coupling_operator: ndarray
        The system operator to which the bath is coupling to.
    spectral_density: BaseSD
        The bath's spectral density.
    temperature: float
        The bath's temperature.
    name: str
        An optional name for the bath.
    description: str
        An optional description of the bath.
    description_dict: dict
        An optional dictionary with descriptive data.

    Raises
    ------
    ValueError:
        If the temperature :math:`T` is smaller then 0.
    NotImplementedError:
        If the coupling_operator is not diagonal.
    """
    def __init__(
            self,
            coupling_operator: ArrayLike,
            spectral_density: BaseSD,
            temperature: Optional[float] = 0.0,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a Bath object. """
        # input check for coupling_operator.
        try:
            __coupling_operator = array(coupling_operator, dtype=NpDtype)
            __coupling_operator.setflags(write=False)
        except:
            raise AssertionError("Coupling operator must be numpy array")
        assert len(__coupling_operator.shape) == 2, \
            "Coupling operator is not a matrix."
        assert __coupling_operator.shape[0] == \
            __coupling_operator.shape[1], \
            "Coupling operator is not a square matrix."
        if not allclose(diag(__coupling_operator.diagonal()),
                        __coupling_operator):
            raise NotImplementedError(
                "Non-diagonal coupling operators are not implemented yet!")
        self._coupling_operator = __coupling_operator
        self._dimension = self._coupling_operator.shape[0]

        # input check for spectral_density.
        if not isinstance(spectral_density, BaseSD):
            raise AssertionError(
                "Spectral density must be an instance of a subclass of BaseSD.")
        self._spectral_density = copy(spectral_density)

        # input check for temperature.
        try:
            __temperature = float(temperature)
        except:
            raise AssertionError("Temperature must be a float.")
        if __temperature < 0.0:
            raise ValueError("Temperature must be >= 0.0 (but is {})".format(
                __temperature))
        self._temperature = __temperature

        super().__init__(name, description, description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dimension     = {} \n".format(self.dimension))
        ret.append("  spectral den. = {} \n".format(self.spectral_density.name))
        ret.append("  temperature   = {} \n".format(self.temperature))
        return "".join(ret)

    @property
    def coupling_operator(self) -> ndarray:
        """The system operator to which the bath couples. """
        return copy(self._coupling_operator)

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension of the coupling operator. """
        return copy(self._dimension)

    @property
    def spectral_density(self) -> BaseSD:
        """The spectral density of the bath. """
        return copy(self._spectral_density)

    @property
    def temperature(self) -> float:
        """The temperature of the bath. """
        return copy(self._temperature)

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
from typing import Callable, Dict, Optional, Text
from copy import copy

from numpy import array, ndarray, diag, allclose

from time_evolving_mpo.spectral_density import BaseSD
from time_evolving_mpo.config import NP_DTYPE, SEPERATOR


class Bath:
    """
    Represents the bath degees of freedom with a specific coupling operator
    (to the system), a specific spectral density and a specific temperature.

    Parameters
    ----------
    coupling_operator: nd.array
        The system operator to which the bath is coupling to.
    spectral_density: BaseSD
        The bath's spectral density.
    temperature: float
        The bath's temperature.
    name: Text
        An optional name for the bath.
    description: Text
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
            coupling_operator: ndarray,
            spectral_density: Callable[[float], float],
            temperature: Optional[float] = 0.0,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a bath object. """
        # input check for coupling_operator.
        try:
            __coupling_operator = array(coupling_operator, dtype=NP_DTYPE)
            __coupling_operator.setflags(write=False)
        except:
            raise AssertionError("Coupling operator must be numpy array")
        assert len(__coupling_operator.shape) == 2, \
            "Coupling operator is not a matrix."
        assert self._coupling_operator.shape[0] == \
            self._coupling_operator.shape[1], \
            "Coupling operator is not a sqare matrix."
        if not allclose(diag(self._coupling_operator.diagonal()),
                        self._coupling_operator):
            raise NotImplementedError(
                "Non-diagonal coupling operators are not implemented yet!")
        self._dimension = coupling_operator.shape[0]

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

        # input check for propeties are in the propertie setters.
        self.name = name
        self.description = description
        self.description_dict = description_dict

    def __str__(self) -> Text:
        ret = []
        ret.append(SEPERATOR)
        ret.append("Bath object: "+self.name+"\n")
        ret.append(" {}\n".format(self.description))
        ret.append(" {}\n".format(self.description_dict))
        return "".join(ret)

    @property
    def coupling_operator(self):
        """The system operator to which the bath couples."""
        return copy(self._coupling_operator)

    @property
    def spectral_density(self):
        """The spectral density of the bath."""
        return copy(self._spectral_density)

    @property
    def temperature(self):
        """The temperature of the bath."""
        return copy(self._temperature)

    @property
    def name(self):
        """Name of the bath."""
        return self._name

    @name.setter
    def name(self, new_name: Text = None):
        if new_name is None:
            new_name = "__unnamed_bath__"
        else:
            assert isinstance(new_name, Text), "Name must be text."
        self._name = new_name

    @name.deleter
    def name(self):
        self.name = None

    @property
    def description(self):
        """Detailed description of the bath."""
        return self._description

    @description.setter
    def description(self, new_description: Text = None):
        if new_description is None:
            new_description = "__no_description__"
        else:
            assert isinstance(new_description, Text), \
                "Description must be text."
        self._description = new_description

    @description.deleter
    def description(self):
        self.description = None

    @property
    def description_dict(self):
        """
        A dictionary for descriptive data.
        """
        return self._description_dict

    @description_dict.setter
    def description_dict(self, new_dict: Dict = None):
        if new_dict is None:
            new_dict = dict({})
        else:
            assert isinstance(new_dict, dict), \
                "Description dictionary must be a dictionary."
        self._description_dict = new_dict

    @description_dict.deleter
    def description_dict(self):
        self.description_dict = None

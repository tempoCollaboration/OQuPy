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
Module for calculating bath dynamics as outlined in [Gribben2021].

**[Gribben2021]**
D. Gribben, A. Strathearn, G. E. Fux, P. Kirton, and B. W. Lovett,
*Using the Environment to Understand non-Markovian Open Quantum Systems*,
arXiv:2106.04212 [quant-ph] (2021).
"""

from typing import Dict, Optional, Text
import numpy as np

from oqupy.base_api import BaseAPIClass
from oqupy.process_tensor import BaseProcessTensor
from oqupy.bath import Bath
from oqupy.system import BaseSystem
from oqupy.config import NpDtype



class TwoTimeBathCorrelations(BaseAPIClass):
    """
    Class to facilitate calculation of two-time bath correlations.
    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The bath object containing all coupling information and temperature.
    process_tensor: ProcessTensor
        The corresponding process tensor calculated for the given bath.
    name: str (default = None)
        An optional name for the bath.
    description: str (default = None)
        An optional description of the bath.
    description_dict: dict (default = None)
        An optional dictionary with descriptive data.
    """
    def __init__(
            self,
            system: BaseSystem,
            bath: Bath,
            process_tensor: BaseProcessTensor,
            system_correlations: Optional[np.ndarray] = np.array([[]],
                                                              dtype=NpDtype),
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None
            ) -> None:
        self._system = system
        self._bath = bath
        self._process_tensor = process_tensor
        self._system_correlations = system_correlations
        self._bath_correlations = {}
        super().__init__(name, description, description_dict)

    @property
    def system(self):
        """
        System Hamiltonian
        """
        return self._system

    @property
    def bath(self):
        """
        Bath properties
        """
        return self._bath

    def bath_occupation(self,
                    freq: float,
                    dw: Optional[float] = 1.0):
        r"""
        Function to calculate the change in bath occupation in a particular
        bandwidth.

        Parameters
        ----------
        freq : float
            Frequency about which to calculate the change in occupation.
        dw : tuple (default = (1.0,1.0)):
            Bandwidth of about the frequency to calculate the energy within. By
            default what is returned by this method is a *density*.
        Returns
        -------
        times : List[float]

        bath_energy: float
        """
        return None

    def correlation(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: Optional[float] = None,
                    time_2: Optional[float] = None,
                    dw: Optional[tuple] = (1.0,1.0),
                    dagg: Optional[tuple] = (1,0)):
        r"""
        Function to calculate two-time correlation function between to frequency
        bands of a bath.

        Parameters
        ----------
        freq_1 : float
            Frequency of the later time operator.
        time_1 : float
            Time the later operator acts.
        freq_2 : float (default = None)
            Frequency of the earlier time operator. If set to None will default
            to freq_2=freq_1.
        time_2 : float (default = None)
            Time the earlier operator acts. If set to None will default to
            time_2=time_1.
        dw : tuple (default = (1.0,1.0)):
            Bandwidth of about each frequency comparing correlations between.
            By default what is returned by this method is a correlation
            *density*.
        dagg : Optional[tuple] (default = (1,0))
            Determines whether each operator is daggered or not e.g. (1,0)
            would correspond to < a^\dagger a >
        Returns
        -------
        times : tuple
            Pair of times of each operation.
        correlation : complex
            Bath correlation function
            <a^{dagg[0]}_{freq_1} (time_1) a^{dagg[1]}_{freq_2} (time_2)>
        """
        return None

    def _calc_kernel(self,
                    freq_1: float,
                    time_1: float,
                    freq_2: float,
                    time_2: float,
                    dagg: tuple):
        r"""
        Function to calculate the corresponding kernel for the desired
        correlation function.
        Parameters
        ----------
        freq_1 : float
            Frequency of the later time operator.
        time_1 : float
            Time the later operator acts.
        freq_2 : float
            Frequency of the earlier time operator.
        time_2 : float
            Time the earlier operator acts.
        dagg : tuple
            Determines whether each operator is daggered or not e.g. (1,0)
            would correspond to < a^\dagger a >

        Returns
        -------
        re_kernel : ndarray
            An array that multiplies the real part of the system correlation
            functions before being summed.
        im_kernel : ndarray
            An array that multiplies the imaginary part of the system
            correlation functions before being summed.

        """
        return None

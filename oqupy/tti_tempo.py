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
Module for the process tensor time evolving matrix product operator algorithm
(PT-TEMPO). This module is based on [Strathearn2018], [Pollock2018],
[Jorgensen2019], and [Fux2021].

**[Strathearn2018]**
A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).

**[Pollock2018]**
F.  A.  Pollock,  C.  Rodriguez-Rosario,  T.  Frauenheim,
M. Paternostro, and K. Modi, *Non-Markovian quantumprocesses: Complete
framework and efficient characterization*, Phys. Rev. A 97, 012127 (2018).

**[Jorgensen2019]**
M. R. Jørgensen and F. A. Pollock, *Exploiting the causal tensor network
structure of quantum processes to efficiently simulate non-markovian path
integrals*, Phys. Rev. Lett. 123, 240602 (2019)

**[Fux2021]**
G. E. Fux, E. Butler, P. R. Eastham, B. W. Lovett, and
J. Keeling, *Efficient exploration of Hamiltonian parameter space for
optimal control of non-Markovian open quantum systems*, Phys. Rev. Lett. 126,
200401 (2021).
"""

from typing import Dict, Optional, Text, Union
from copy import copy

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.bath import Bath
from oqupy.config import PT_DEFAULT_TOLERANCE
from oqupy.config import PT_TEMPO_BACKEND_CONFIG
from oqupy.process_tensor import BaseProcessTensor
from oqupy.process_tensor import SimpleProcessTensor
from oqupy.process_tensor import FileProcessTensor
from oqupy.backends.pt_tempo_backend import PtTempoBackend
from oqupy.tempo import TempoParameters
from oqupy.tempo import guess_tempo_parameters
from oqupy.tempo import influence_matrix
from oqupy.operators import left_right_super
from oqupy.util import get_progress

from oqupy.iTEBD_TEMPO_useoqupybath import iTEBD_TEMPO_oqupy
from oqupy.process_tensor import TTInvariantProcessTensor

class TTITempo():
    """
    Class to facilitate a PT-TEMPO computation with time-translation invariant process tensor

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the PT-TEMPO computation.
    start_time: float
        The start time.
    unique: bool (default = False),
        Whether to use degeneracy checking. If True reduces dimension of
        bath tensors in case of degeneracies in sums ('west') and
        sums,differences ('north') of the bath coupling operator.
        See bath:north_degeneracy_map, bath:west_degeneracy_map.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    """
    def __init__(
            self,
            bath: Bath,
            start_time: float,
            parameters: TempoParameters,
            rank: Optional[int] = np.inf,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a PtTempo object. """
        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath
        self._dimension = self._bath.dimension
        self._correlations = self._bath.correlations

        # super().__init__(name, description)

        try:
            tmp_start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = tmp_start_time

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        self._rank=rank # Passed to the iTEBD code as the maximum rank.
        
        self._name=name
        self._description=description

        self._init_tti_process_tensor()

        self._coupling_comm = self._bath._coupling_comm
        self._coupling_acomm = self._bath._coupling_acomm

        self._backend_instance = None

    def _init_tti_process_tensor(self):
        """ToDo. """
        unitary = self._bath.unitary_transform
        if not np.allclose(unitary, np.identity(self._dimension)):
            transform_in = left_right_super(unitary.conjugate().T,
                                            unitary).T
            transform_out = left_right_super(unitary,
                                             unitary.conjugate().T).T
        else:
            transform_in = None
            transform_out = None
        
        MyiTEBD_TEMPO_oqupy = iTEBD_TEMPO_oqupy(np.diagonal(self._bath._coupling_operator), self._parameters.dt, 
                                                self._bath.correlations, self._parameters.dkmax)
        MyiTEBD_TEMPO_oqupy.compute_f(self._parameters.epsrel,self._rank)
        
        self._process_tensor = TTInvariantProcessTensor(MyiTEBD_TEMPO_oqupy,
            transform_in=transform_in,
            transform_out=transform_out,
            name=self._name,
            description=self._description)
        
    def get_process_tensor(self):
        return self._process_tensor
    



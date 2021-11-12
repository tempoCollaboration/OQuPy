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
        return None

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
Shorthand for commonly used operators.
"""

from typing import Text

import numpy as np

from time_evolving_mpo.config import NP_DTYPE

PAULI = {"id":[[1, 0], [0, 1]],
         "x":[[0, 1], [1, 0]],
         "y":[[0, -1j], [1j, 0]],
         "z":[[1, 0], [0, -1]]}

def identity(n: int):
    """
    Identity matrix of dimension `n` x `n`.

    Parameters
    ----------
    n: int
        dimension of the square matrix

    Returns
    -------
    The n x n identity matrix. : np.array
    """
    return np.identity(n, dtype=NP_DTYPE)

def pauli(name: Text):
    """
    Pauli matrix of type `name`.

    Parameters
    ----------
    name: str{ ``'id'``, ``'x'``, ``'y'``, ``'z'``}

    Returns
    -------
    The  pauli matrix. : np.array
    """
    return np.array(PAULI[name], dtype=NP_DTYPE)

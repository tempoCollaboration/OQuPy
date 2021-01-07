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
Shorthand for commonly used operators.
"""

from typing import Text

import numpy as np

from time_evolving_mpo.config import NpDtype

SIGMA = {"id":[[1, 0], [0, 1]],
         "x":[[0, 1], [1, 0]],
         "y":[[0, -1j], [1j, 0]],
         "z":[[1, 0], [0, -1]],
         "+":[[0, 1], [0, 0]],
         "-":[[0, 0], [1, 0]]}

SPIN_DM = {"up":[[1, 0], [0, 0]],
           "down":[[0, 0], [0, 1]],
           "z+":[[1, 0], [0, 0]],
           "z-":[[0, 0], [0, 1]],
           "x+":[[0.5, 0.5], [0.5, 0.5]],
           "x-":[[0.5, -0.5], [-0.5, 0.5]],
           "y+":[[0.5, -0.5j], [0.5j, 0.5]],
           "y-":[[0.5, 0.5j], [-0.5j, 0.5]],
           "mixed":[[0.5, 0.0], [0.0, 0.5]]}


def identity(n: int):
    """
    Identity matrix of dimension `n` x `n`.

    Parameters
    ----------
    n: int
        Dimension of the square matrix.

    Returns
    -------
    identity : ndarray
        Identity matrix of dimension `n` x `n`.
    """
    return np.identity(n, dtype=NpDtype)


def sigma(name: Text):
    """
    Spin matrix sigma of type `name`.

    Parameters
    ----------
    name: str{ ``'id'``, ``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``}

    Returns
    -------
    sigma : ndarray
        Spin matrix of type `name`.
    """
    return np.array(SIGMA[name], dtype=NpDtype)


def spin_dm(name: Text):
    """
    Spin 1/2 state of type `name`.

    Parameters
    ----------
    name: str{ ``'up'``/``'z+'``, ``'down'``/``'z-'``, ``'x+'``, ``'x-'``, \
    ``'y+'``, ``'y-'``, ``mixed``}

    Returns
    -------
    density_matrix : ndarray
        Spin density matrix.
    """
    return np.array(SPIN_DM[name], dtype=NpDtype)


def create(n: int):
    """
    Bosonic creation operator of dimension `n` x `n`.

    Parameters
    ----------
    n: int
        Dimension of the Hilbert space.

    Returns
    -------
    create : ndarray
        Creation operator matrix of dimension `n` x `n`.
    """
    return destroy(n).T


def destroy(n: int):
    """
    Bosonic annihilation operator of dimension `n` x `n`.

    Parameters
    ----------
    n: int
        Dimension of the Hilbert space.

    Returns
    -------
    create : ndarray
        Annihilation operator matrix of dimension `n` x `n`.
    """
    return np.diag(np.sqrt(range(1, n), dtype=NpDtype), 1)

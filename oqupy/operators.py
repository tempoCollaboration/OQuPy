# Copyright 2022 The TEMPO Collaboration
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
from numpy import ndarray

from oqupy.config import NpDtype


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


def identity(n: int) -> ndarray:
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


def sigma(name: Text) -> ndarray:
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


def spin_dm(name: Text) -> ndarray:
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


def create(n: int) -> ndarray:
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


def destroy(n: int) -> ndarray:
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


# -- superoperators ----------------------------------------------------------

def commutator(operator: ndarray) -> ndarray:
    """Construct commutator superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            - np.kron(np.identity(dim), operator.T)

def acommutator(operator: ndarray) -> ndarray:
    """Construct anti-commutator superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            + np.kron(np.identity(dim), operator.T)

def left_super(operator: ndarray) -> ndarray:
    """Construct left acting superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim))

def right_super(operator: ndarray) -> ndarray:
    """Construct right acting superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(np.identity(dim), operator.T)

def left_right_super(
        left_operator: ndarray,
        right_operator: ndarray) -> ndarray:
    """Construct left and right acting superoperator from operators. """
    return np.kron(left_operator, right_operator.T)

def preparation(
        density_matrix: ndarray) -> ndarray:
    """Construct the super operator that prepares the state. """
    dim = density_matrix.shape[0]
    identity_matrix = np.identity(dim, dtype=NpDtype)
    return np.outer(density_matrix.flatten(), identity_matrix.flatten())


# -- two site superoperators --------------------------------------------------

def cross_commutator(
        operator_1: ndarray,
        operator_2: ndarray) -> ndarray:
    """Construct commutator of cross term (acting on two Hilbert spaces). """
    id1 = np.identity(operator_1.shape[1])
    id2 = np.identity(operator_2.shape[1])
    op1_id = np.kron(operator_1, id1)
    op2_id = np.kron(operator_2, id2)
    id_op1 = np.kron(id1, operator_1.T)
    id_op2 = np.kron(id2, operator_2.T)
    return np.kron(op1_id, op2_id) - np.kron(id_op1, id_op2)

def cross_acommutator(
        operator_1: ndarray,
        operator_2: ndarray) -> ndarray:
    """
    Construct anit-commutator of cross term (acting on two Hilbert spaces).
    """
    id1 = np.identity(operator_1.shape[1])
    id2 = np.identity(operator_2.shape[1])
    op1_id = np.kron(operator_1, id1)
    op2_id = np.kron(operator_2, id2)
    id_op1 = np.kron(id1, operator_1.T)
    id_op2 = np.kron(id2, operator_2.T)
    return np.kron(op1_id, op2_id) + np.kron(id_op1, id_op2)

def cross_left_right_super(
        operator_1_l: ndarray,
        operator_1_r: ndarray,
        operator_2_l: ndarray,
        operator_2_r: ndarray) -> ndarray:
    """
    Construct anit-commutator of cross term (acting on two Hilbert spaces).
    """
    op1l_op1r = np.kron(operator_1_l, operator_1_r.T)
    op2l_op2r = np.kron(operator_2_l, operator_2_r.T)
    return np.kron(op1l_op1r, op2l_op2r)

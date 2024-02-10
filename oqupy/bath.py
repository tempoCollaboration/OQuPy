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
Module on physical information on the bath and its coupling to the system.
"""
from typing import Optional, Text
from copy import copy

import numpy as np
from numpy import ndarray

from oqupy.config import NpDtype
from oqupy.config import DEFAULT_TOLERANCE_DEGENERACY
from oqupy.correlations import BaseCorrelations
from oqupy.base_api import BaseAPIClass
from oqupy.operators import commutator, acommutator


class Bath(BaseAPIClass):
    """
    Represents the bath degrees of freedom with a specific coupling operator
    (to the system) and a specific auto-correlation function.

    Parameters
    ----------
    coupling_operator: np.ndarray
        The system operator to which the bath couples.
    correlations: BaseCorrelations
        The bath's auto correlation function.
    name: str
        An optional name for the bath.
    description: str
        An optional description of the bath.
    """
    def __init__(
            self,
            coupling_operator: ndarray,
            correlations: BaseCorrelations,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Creates a Bath object. """
        # input check for coupling_operator.
        try:
            tmp_coupling_operator = np.array(coupling_operator, dtype=NpDtype)
            tmp_coupling_operator.setflags(write=False)
        except Exception as e:
            raise AssertionError("Coupling operator must be numpy array") \
                from e
        assert len(tmp_coupling_operator.shape) == 2, \
            "Coupling operator must be a matrix."
        assert tmp_coupling_operator.shape[0] == \
            tmp_coupling_operator.shape[1], \
            "Coupling operator must be a square matrix."
        assert np.allclose(tmp_coupling_operator.conjugate().T,
                           tmp_coupling_operator), \
            "Coupling operator must be a hermitian matrix."
        self._dimension = tmp_coupling_operator.shape[0]

        # diagonalise the coupling operator
        if np.allclose(np.diag(tmp_coupling_operator.diagonal()),
                        tmp_coupling_operator):
            self._coupling_operator = tmp_coupling_operator
            self._unitary = np.identity(self._dimension)
        else:
            w, v = np.linalg.eig(tmp_coupling_operator)
            self._coupling_operator = np.diag(w)
            self._unitary = v
            assert np.allclose(tmp_coupling_operator, \
                self._unitary @ self._coupling_operator \
                @ self._unitary.conjugate().T)

        # identify degeneracies in eigensystem of coupling operator
        tmp_coupling_comm = commutator(self._coupling_operator)
        tmp_coupling_acomm = acommutator(self._coupling_operator)
        self._coupling_comm = tmp_coupling_comm.diagonal()
        self._coupling_acomm = tmp_coupling_acomm.diagonal()

        self._north_degeneracy_map = _row_degeneracy([self._coupling_comm,
                                                      self._coupling_acomm])
        self._west_degeneracy_map = _row_degeneracy([self._coupling_comm])

        # input check for correlations.
        if not isinstance(correlations, BaseCorrelations):
            raise AssertionError(
                "Correlations must be an instance of a subclass of " \
                + "BaseCorrelations.")
        self._correlations = copy(correlations)

        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dimension     = {} \n".format(self.dimension))
        ret.append("  correlations  = {} \n".format(self.correlations.name))
        return "".join(ret)

    @property
    def coupling_operator(self) -> np.ndarray:
        """The diagonalised system operator to which the bath couples. """
        return self._coupling_operator.copy()

    @property
    def unitary_transform(self) -> np.ndarray:
        """The unitary that makes the coupling operator diagonal. """
        return self._unitary.copy()

    @property
    def dimension(self) -> np.ndarray:
        """Hilbert space dimension of the coupling operator. """
        return copy(self._dimension)

    @property
    def correlations(self) -> BaseCorrelations:
        """The correlations of the bath. """
        return copy(self._correlations)

    @property
    def coupling_acomm(self) -> np.ndarray:
        """Diagonal elements of the anti-commutator of the coupling
        operator. """
        return self._coupling_acomm.copy()

    @property
    def coupling_comm(self) -> np.ndarray:
        """Diagonal elements of the commutator of the coupling
        operator. """
        return self._coupling_comm.copy()

    @property
    def north_degeneracy_map(self) -> np.ndarray:
        """Map to minimal set of indices for influence tensors in
        north-south direction according to simultaneous degeneracies in
        sums & differences of eigenvalues of coupling operator (minimal
        dimension is number of unique values in this map).
        Used by a Tempo computation if unique==True only. """
        return copy(self._north_degeneracy_map)

    @property
    def west_degeneracy_map(self) -> np.ndarray:
        """Map to minimal set of indices for influence tensors in
        west-east direction according to degeneracies in sums of
        eigenvalues of coupling operator (minimal dimension is number
        of unique values in this map).
        Used by a Tempo computation if unique==True only. """
        return copy(self._west_degeneracy_map)


def _row_degeneracy(matrix):
    """Finds the row degeneracy of matrix. Returns array of
    indices mapping full space to non-degenerate rows (repeated
    indices indicate row degeneracy in the original matrix)."""
    mat = np.array(matrix).round(decimals=DEFAULT_TOLERANCE_DEGENERACY)
    return_map = np.unique(mat.T,return_inverse=True,axis=0)[1]
    return return_map

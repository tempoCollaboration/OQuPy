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
Module for tensor network backend.
"""

from typing import Callable, Dict, Tuple
from copy import copy

from numpy import ndarray, array

from time_evolving_mpo.config import NP_DTYPE
from time_evolving_mpo.backends.tensor_network import mps_mpo as mm
from time_evolving_mpo.backends.base_backend import BaseBackend
from time_evolving_mpo.backends.base_backend import BaseTempoBackend
from time_evolving_mpo.backends.tensor_network.util import create_delta
from time_evolving_mpo.backends.tensor_network.util import add_singleton


MPS_SINGLETON = array([[[1.0]]], dtype=NP_DTYPE)


class TensorNetworkTempoBackend(BaseTempoBackend):
    """See BaseTempoBackend for docstring. """
    def __init__(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float):
        """Create a TensorNetworkTempoBackend object. """
        self._initial_state = initial_state
        self._influence = influence
        self._propagators = propagators
        self._sum_north = sum_north
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._step = None
        self._state = None
        self._grow = None
        self._mps = None
        self._mpo = None

    def initialize(self) -> Tuple[int, ndarray]:
        """See BaseBackend.initialize() for docstring."""
        initial_tensor = copy(self._initial_state)
        initial_tensor.shape = (1, *initial_tensor.shape, 1)

        self._mps = mm.SimpleAgnosticMPS([initial_tensor],
                                         name="theeMPS")
        self._mpo = mm.SimpleAgnosticMPO([],
                                         name="coreMPO")

        self._step = 0
        self._grow = True
        self._state = self._initial_state
        return self._step, copy(self._state)

    def compute_step(self) -> Tuple[int, ndarray]:
        """See BaseBackend.compute_step() for docstring."""

        prop_1, prop_2 = self._propagators(self._step)
        prop_2_with_singletons = add_singleton(add_singleton(prop_2, 1), 3)

        if self._step == self._dkmax:
            infl = self._influence(self._step)
            infl_three_legs = add_singleton(create_delta(infl, [0, 0, 1]), 0)
            self._mpo.append_left(infl_three_legs)
            self._grow = False

        mpo = self._mpo.copy()
        mpo.name = "tmp mpo"
        if self._grow:
            infl = self._influence(self._step)
            infl_three_legs = add_singleton(create_delta(infl, [0, 0, 1]), 0)
            infl_four_legs = create_delta(infl, [1, 0, 0, 1])
            mpo.append_left(infl_three_legs)
            self._mpo.append_left(infl_four_legs)
        mpo.append_right(prop_2_with_singletons)

        self._mps.contract_with_matrix(prop_1, -1)
        self._mps.append_right(MPS_SINGLETON)

        self._mps.update_node_names("mps{}step".format(self._step))
        # print("contract step {}:".format(self._step), flush=True)
        # print(self._mps)

        self._mps.contract_with_mpo(
            mpo,
            from_index=0,
            to_index=len(self._mps)-1,
            max_truncation_err=self._epsrel,
            relative=True)

        if not self._grow:
            self._mps.contract_with_vectors(
                [self._sum_north],
                from_index=0,
                to_index=1)

        self._mps.svd_sweep(
            from_index=len(self._mps)-1,
            to_index=0,
            max_truncation_err=self._epsrel,
            relative=True)

        mps = self._mps.copy()
        mps.contract_with_vectors(
            [self._sum_north]*(len(mps)-1),
            from_index=0,
            to_index=len(mps)-1)
        self._state = mps.get_single()
        del mps

        self._step = self._step + 1
        return self._step, copy(self._state)


class TensorNetworkBackend(BaseBackend):
    """See BaseBackend for docstring. """
    def __init__(self, config: Dict) -> None:
        """Create ExampleBackend object. """
        self._tempo_backend_class = TensorNetworkTempoBackend

    def get_tempo_backend(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float) -> BaseTempoBackend:
        """Returns an TensorNetworkTempoBackend object. """
        return self._tempo_backend_class(initial_state,
                                         influence,
                                         propagators,
                                         sum_north,
                                         sum_west,
                                         dkmax,
                                         epsrel)

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

from numpy import ndarray, array, moveaxis, dot

from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.backends.tensor_network import node_array as na
from time_evolving_mpo.backends.base_backend import BaseBackend
from time_evolving_mpo.backends.base_backend import BaseTempoBackend
from time_evolving_mpo.backends.tensor_network.util import create_delta
import time_evolving_mpo.util as util

MPS_SINGLETON = array([[[1.0]]], dtype=NpDtype)


class TensorNetworkTempoBackend(BaseTempoBackend):
    """See BaseTempoBackend for docstring. """
    def __init__(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            unitary_transform: ndarray,
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float):
        """Create a TensorNetworkTempoBackend object. """
        super().__init__(initial_state,
                         influence,
                         unitary_transform,
                         propagators,
                         sum_north,
                         sum_west,
                         dkmax,
                         epsrel)
        self._grow = None
        self._mps = None
        self._mpo = None
        self._super_u = None
        self._super_u_dagg = None
        self._sum_north_na = None
        self._influences = None

    def initialize(self) -> Tuple[int, ndarray]:
        """See BaseBackend.initialize() for docstring."""
        self._initial_state = copy(self._initial_state).reshape(-1)

        self._super_u = util.left_right_super(
                            self._unitary_transform,
                            self._unitary_transform.conjugate().T)
        self._super_u_dagg = util.left_right_super(
                                self._unitary_transform.conjugate().T,
                                self._unitary_transform)

        self._sum_north_na = na.NodeArray([self._sum_north],
                                          left=False,
                                          right=False,
                                          name="Sum north")

        influences = []
        if self._dkmax is None:
            dkmax_pre_compute = 1
        else:
            dkmax_pre_compute = self._dkmax

        for i in range(dkmax_pre_compute):
            infl = self._influence(i)
            infl_four_legs = create_delta(infl, [1, 0, 0, 1])
            if i == 0:
                tmp = dot(moveaxis(infl_four_legs,1,-1),
                          self._super_u_dagg)
                tmp = moveaxis(tmp,-1,1)
                tmp = dot(tmp, self._super_u.T)
                infl_four_legs = tmp
            influences.append(infl_four_legs)

        self._mps = na.NodeArray([self._initial_state],
                                 left=False,
                                 right=False,
                                 name="Thee MPS")
        self._mpo = na.NodeArray(list(reversed(influences)),
                                 left=True,
                                 right=True,
                                 name="Thee Time Evolving MPO")

        self._step = 0
        self._state = self._initial_state

        return self._step, copy(self._state)

    def compute_step(self) -> Tuple[int, ndarray]:
        """
        See BaseBackend.compute_step() for docstring.

        For example, for at step 4, we start with:

            A ... self._mps
            B ... self._mpo
            w ... self._sum_west
            n ... self._sum_north_array
            p1 ... prop_1
            p2 ... prop_2

              n  n  n  n
              |  |  |  |

              |  |  |  |     |
        w~~ ~~B~~B~~B~~B~~ ~~p2
              |  |  |  |
                       p1
              |  |  |  |
              A~~A~~A~~A

        return:
            step = 4
            state = contraction of A,B,w,n,p1

        effects:
            self._mpo will grow to the left with the next influence functional
            self._mps will be contraction of A,B,w,p1,p2

        """
        self._step += 1
        prop_1, prop_2 = self._propagators(self._step)
        prop_1_na = na.NodeArray([prop_1],
                                 left=False,
                                 right=False,
                                 name="first half-step")
        prop_2_na = na.NodeArray([prop_2],
                                 left=True,
                                 right=False,
                                 name="second half-step")

        if self._dkmax is None:
            mpo = self._mpo.copy()
            infl = self._influence(len(mpo))
            infl_four_legs = create_delta(infl, [1, 0, 0, 1])
            infl_na = na.NodeArray([infl_four_legs],
                                   left=True,
                                   right=True)
            self._mpo = na.join(infl_na,
                                self._mpo,
                                name="Thee Time Evolving MPO",
                                copy=False)
        elif self._step < self._dkmax:
            _, mpo = na.split(self._mpo,
                              int(0 - self._step),
                              copy=True)
        else:
            mpo = self._mpo.copy()

        mpo.name = "temporary MPO"
        mpo.apply_vector(self._sum_west, left=True)

        self._mps.zip_up(prop_1_na,
                         axes=[(0,0)],
                         left_index=-1,
                         right_index=-1,
                         direction="left",
                         max_singular_values=None,
                         max_truncation_err=self._epsrel,
                         relative=True,
                         copy=False)

        self._mps.zip_up(mpo,
                         axes=[(0, 0)],
                         left_index=0,
                         right_index=-1,
                         direction="right",
                         max_singular_values=None,
                         max_truncation_err=self._epsrel,
                         relative=True,
                         copy=False)

        if self._dkmax is not None and self._step >= self._dkmax:
            self._mps.contract(self._sum_north_na,
                               axes=[(0,0)],
                               left_index=0,
                               right_index=0,
                               direction="right",
                               copy=True)

        self._mps.svd_sweep(from_index=-1,
                            to_index=0,
                            max_singular_values=None,
                            max_truncation_err=self._epsrel,
                            relative=True)

        self._mps = na.join(self._mps,
                            prop_2_na,
                            copy=False,
                            name=f"Thee MPS ({self._step})")

        tmp_mps = self._mps.copy()
        for _ in range(len(tmp_mps)-1):
            tmp_mps.contract(self._sum_north_na,
                             axes=[(0,0)],
                             left_index=0,
                             right_index=0,
                             direction="right",
                             copy=True)

        assert len(tmp_mps) == 1
        assert not tmp_mps.left
        assert not tmp_mps.right
        assert tmp_mps.rank == 1
        self._state = tmp_mps.nodes[0].get_tensor()

        return copy(self._step), copy(self._state)

class TensorNetworkBackend(BaseBackend):
    """See BaseBackend for docstring. """
    def __init__(self, config: Dict) -> None:
        """Create ExampleBackend object. """
        self._tempo_backend_class = TensorNetworkTempoBackend

    def get_tempo_backend(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            unitary_transform: ndarray,
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float) -> BaseTempoBackend:
        """Returns an TensorNetworkTempoBackend object. """
        return self._tempo_backend_class(initial_state,
                                         influence,
                                         unitary_transform,
                                         propagators,
                                         sum_north,
                                         sum_west,
                                         dkmax,
                                         epsrel)

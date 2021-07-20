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
Module for tensor network process tensor tempo backend.
"""

from typing import Callable, Dict

import numpy as np
from numpy import ndarray

from time_evolving_mpo.backends.tensor_network import node_array as na
from time_evolving_mpo.backends.base_backends import BasePtTempoBackend
from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.process_tensor import BaseProcessTensor
from time_evolving_mpo import util


class TensorNetworkPtTempoBackend(BasePtTempoBackend):
    """See BasePtTempoBackend for docstring. """
    def __init__(
            self,
            dimension: int,
            influence: Callable[[int], ndarray],
            process_tensor: BaseProcessTensor,
            sum_north: ndarray,
            sum_west: ndarray,
            num_steps: int,
            dkmax: int,
            epsrel: float,
            config: Dict):
        """Create a TensorNetworkPtTempoBackend object. """
        super().__init__(dimension,
                         influence,
                         process_tensor,
                         sum_north,
                         sum_west,
                         num_steps,
                         dkmax,
                         epsrel,
                         config)
        if "backend" in config:
            self._backend = config["backend"]
        else:
            self._backend = None

        self._mps = None
        self._mpo = None
        self._last_infl_na = None
        self._num_infl = min(num_steps, dkmax+1)
        self._super_u = None
        self._super_u_dagg = None
        self._sum_north_scaled = None
        self._one_na = None


    def initialize(self) -> None:
        """Initializes the PT-TEMPO tensor network. """
        # create mpo
        # create mpo last
        # copy and contrac mpo to mps
        scale = self._dimension

        self._sum_north_scaled = self._sum_north * scale

        influences_mpo = []
        influences_mps = []
        for i in range(self._num_infl):
            if i == 0:
                infl = self._influence(i)
                infl = infl / scale
                infl_mpo = util.create_delta(infl, [1, 1, 0])
                infl_mps = infl.T / scale
            elif i == self._num_infl-1:
                if self._dkmax < self._num_steps:
                    infl = self._influence(-1)
                else:
                    infl = self._influence(i)
                infl_mpo = util.add_singleton(infl, 1)
                infl_mpo = util.add_singleton(infl_mpo, 3)
                infl_mps = util.add_singleton(infl, 2)
            else:
                infl = self._influence(i)
                infl_mpo = util.create_delta(infl, [0, 1, 1, 0])
                infl_mps = util.create_delta(infl / scale, [0, 1, 0])

            influences_mpo.append(infl_mpo)
            influences_mps.append(infl_mps)



        self._mpo = na.NodeArray(influences_mpo,
                                 left=False,
                                 right=True,
                                 name="Thee Time Evolving MPO",
                                 backend=self._backend)

        self._mps = na.NodeArray(influences_mps,
                                 left=False,
                                 right=True,
                                 name="Thee MPS",
                                 backend=self._backend)

        self._mps.svd_sweep(from_index=-1,
                            to_index=0,
                            max_singular_values=None,
                            max_truncation_err=self._epsrel,
                            relative=True)

        self._mps.svd_sweep(from_index=0,
                            to_index=-1,
                            max_singular_values=None,
                            max_truncation_err=self._epsrel,
                            relative=True)

        one = np.array([[1.0]], dtype=NpDtype)
        self._one_na = na.NodeArray([one],
                                   left=True,
                                   right=False,
                                   name="The id",
                                   backend=self._backend)

        self._step = 1

    def compute_step(self) -> None:
        """
        See BasePtBackend.compute_step() for docstring.

        For example, for at step 4, we start with:

            A ... self._mps
            B ... self._mpo
            1 ... [[1.0]]
            n ... self._sum_north_array

                                n
                |               |
            1~ ~B~          A~ ~B~
            |   |           |   |
            A~ ~B~          A~ ~B~
            |   |           |   |
            A~ ~B~    or    A~ ~B~
            |   |           |   |
            A~ ~B~          A~ ~B~
            |               |
            A~              A~
            |               |
            A~              A~


        effects:
            if in grow phase:
                self._mps will be contraction of A, B, 1
            if in end phase:
                self._mps will be a contraction of A, B, n
                self._mpo will be one element shorter
        """
        self._step += 1

        end_phase = bool(self._step > self._num_steps - self._num_infl + 1)

        if end_phase:
            self._mpo, _ = na.split(self._mpo,
                                    index=-1,
                                    copy=False,
                                    name_left="Shortened MPO")
            self._mpo.apply_vector(self._sum_north_scaled, left=False)
            if self._mps.right:
                self._mps.apply_vector(np.array([1.0]), left=False)
        else:
            if self._dkmax is not None:
                dk = int(0 - self._step)
                infl = self._influence(dk)
                infl_mpo = util.add_singleton(infl, 1)
                infl_mpo = util.add_singleton(infl_mpo, 3)
                last_mpo = na.NodeArray(
                        [infl_mpo],
                        left=True,
                        right=True,
                        name="The integrating back to zero infl. func.",
                        backend=self._backend)
                self._mpo, _ = na.split(self._mpo,
                                        index=-1,
                                        copy=False,
                                        name_left="Shortened MPO")
                self._mpo = na.join(self._mpo,
                                    last_mpo,
                                    copy=False,
                                    name="Thee updated MPO")

            one = self._one_na.copy()
            self._mps = na.join(self._mps,
                                one,
                                copy=False,
                                name="Thee updated MPS")

        mpo = self._mpo.copy()

        self._mps.zip_up(mpo,
                         axes=[(0,0)],
                         right_index=-1,
                         direction="left",
                         max_singular_values=None,
                         max_truncation_err=self._epsrel,
                         relative=True,
                         copy=False)

        self._mps.svd_sweep(from_index=self._step-2,
                            to_index=-1,
                            max_singular_values=None,
                            max_truncation_err=self._epsrel,
                            relative=True)

        return self._step < self._num_steps

    def get_mpo_tensor(self, step: int) -> ndarray:
        """ToDo. """
        n = len(self._mps.nodes)
        assert n == self._num_steps
        assert step < n

        if step == 0:
            order = [self._mps.bond_edges[0],self._mps.array_edges[0][0]]
            first_t = self._mps.nodes[0].reorder_edges(order).get_tensor()
            first_t = util.add_singleton(first_t, 0)
            tensor = first_t * self._dimension
        elif step == n-1:
            order = [self._mps.bond_edges[-1],self._mps.array_edges[-1][0]]
            last_t = self._mps.nodes[-1].reorder_edges(order).get_tensor()
            last_t = util.add_singleton(last_t, 1)
            tensor = last_t * self._dimension
        else:
            order = [self._mps.bond_edges[step-1],
            self._mps.bond_edges[step],
            self._mps.array_edges[step][0]]
            temp_t = self._mps.nodes[step].reorder_edges(order).get_tensor()
            tensor = temp_t * self._dimension

        return tensor

    def update_process_tensor(self) -> None:
        """Update the process tensor. """
        assert self._step >= self._num_steps

        for step in reversed(range(self.num_steps)):
            mpo_tensor = self.get_mpo_tensor(step)
            self._process_tensor.set_mpo_tensor(step, mpo_tensor)
        self._process_tensor.compute_caps()

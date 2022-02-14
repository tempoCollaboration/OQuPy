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
Module for tempo backend.
"""

from typing import Callable, Dict, Tuple
from copy import copy

from numpy import ndarray, moveaxis, dot

from oqupy.tempo.backends import node_array as na
from oqupy.util import create_delta
from oqupy import operators


class TempoBackend:
    """
    Backend class for TEMPO.

    Parameters
    ----------
    initial_state: ndarray
        The initial density matrix (as a vector).
    influence: callable(int) -> ndarray
        Callable that takes an integer `step` and returns the influence super
        operator of that `step`.
    unitary_transform: ndarray
        Unitary that transforms the coupling operator into a diagonal form.
    propagators: callable(int) -> ndarray, ndarray
        Callable that takes an integer `step` and returns the first and second
        half of the system propagator of that `step`.
    sum_north: ndarray
        The summing vector for the north leggs.
    sum_west: ndarray
        The summing vector for the west leggs.
    dkmax: int
        Number of influences to include. If ``dkmax == None`` then all
        influences are included.
    epsrel: float
        Maximal relative SVD truncation error.
    """
    def __init__(
            self,
            initial_state: ndarray,
            influence: Callable[[int], ndarray],
            unitary_transform: ndarray,
            propagators: Callable[[int], Tuple[ndarray, ndarray]],
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float,
            config: Dict):
        """Create a TempoBackend object. """
        self._initial_state = initial_state
        self._influence = influence
        self._unitary_transform = unitary_transform
        self._propagators = propagators
        self._sum_north = sum_north
        self._sum_west = sum_west
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._step = None
        self._state = None
        self._config = config
        self._mps = None
        self._mpo = None
        self._super_u = None
        self._super_u_dagg = None
        self._sum_north_na = None

    @property
    def step(self) -> int:
        """The current step in the TEMPO computation. """
        return self._step

    def initialize(self) -> Tuple[int, ndarray]:
        """
        Initializes the TEMPO tensor network.

        Returns
        -------
        step: int = 0
            The current step count, which after initialization, is 0 .
        state: ndarray
            Density matrix (as a vector) at the current step, which after
            initialization, is just the initial state.
        """
        self._initial_state = copy(self._initial_state).reshape(-1)

        self._super_u = operators.left_right_super(
                                self._unitary_transform,
                                self._unitary_transform.conjugate().T)
        self._super_u_dagg = operators.left_right_super(
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
            dkmax_pre_compute = self._dkmax + 1

        for i in range(dkmax_pre_compute):
            infl = self._influence(i)
            infl_four_legs = create_delta(infl, [1, 0, 0, 1])
            if i == 0:
                tmp = dot(moveaxis(infl_four_legs, 1, -1),
                          self._super_u_dagg)
                tmp = moveaxis(tmp, -1, 1)
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
        Takes a step in the TEMPO tensor network computation.

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

        Returns
        -------
        step: int
            The current step count.
        state: ndarray
            Density matrix at the current step.

        """
        self._step += 1
        prop_1, prop_2 = self._propagators(self._step-1)
        prop_1_na = na.NodeArray([prop_1.T],
                                 left=False,
                                 right=False,
                                 name="first half-step")
        prop_2_na = na.NodeArray([prop_2.T],
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
        elif self._step <= self._dkmax:
            _, mpo = na.split(self._mpo,
                              int(0 - self._step),
                              copy=True)
        elif self._step == self._dkmax:
            mpo = self._mpo.copy()
        else: # self._step >= self._dkmax
            mpo = self._mpo.copy()
            infl = self._influence(self._dkmax-self._step)
            if infl is not None:
                infl_four_legs = create_delta(infl, [1, 0, 0, 1])
                infl_na = na.NodeArray([infl_four_legs],
                                       left=True,
                                       right=True)
                _, mpo = na.split(self._mpo,
                                  index=1,
                                  copy=True)
                mpo = na.join(infl_na,
                              mpo,
                              name="Thee Time Evolving MPO",
                              copy=False)

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

        if len(self._mps) != len(mpo):
            self._mps.contract(self._sum_north_na,
                               axes=[(0,0)],
                               left_index=0,
                               right_index=0,
                               direction="right",
                               copy=True)

        self._mps.zip_up(mpo,
                         axes=[(0, 0)],
                         left_index=0,
                         right_index=-1,
                         direction="right",
                         max_singular_values=None,
                         max_truncation_err=self._epsrel,
                         relative=True,
                         copy=False)

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

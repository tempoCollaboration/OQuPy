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

from typing import Callable, Dict, Optional, List

from numpy import ndarray

from oqupy.backends import node_array as na
from oqupy.process_tensor import BaseProcessTensor
from oqupy.util import add_singleton, create_delta

from oqupy.backends.numerical_backend import np

class PtTempoBackend:
    """
    Base class for process tensor tempo backends.

    Parameters
    ----------
    influence: callable(int) -> ndarray
        Callable that takes an integer `step` and returns the influence super
        operator of that `step`.
    process_tensor: BaseProcessTensor
        Todo
    sum_north: ndarray
        The summing vector for the north leggs.
    sum_west: ndarray
        The summing vector for the west leggs.
    num_steps: int
        Todo
    dkmax: int
        Number of influences to include. If ``dkmax == None`` then all
        influences are included.
    epsrel: float
        Maximal relative SVD truncation error.
    config: Dict
        Todo
    degeneracy_maps: Optional[List[ndarray]] (default None)
        List of two arrays to invert the degeneracy in the north and west
        directions (respectively). If None then no degeneracy checks are used.
    """
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
            config: Dict,
            degeneracy_maps: Optional[List[ndarray]] = None):
        """Create a BasePtTempoBackend object. """
        self._dimension = dimension
        self._influence = influence
        self._process_tensor = process_tensor
        self._sum_north = sum_north
        self._sum_west = sum_west
        self._num_steps = num_steps
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._config = config
        self._step = None
        self._degeneracy_maps = degeneracy_maps

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

    @property
    def step(self) -> int:
        """The current step in the PT-TEMPO computation. """
        return self._step

    @property
    def num_steps(self) -> int:
        """The current step in the PT-TEMPO computation. """
        return self._num_steps

    def initialize(self) -> None:
        """Initializes the PT-TEMPO tensor network. """
        # create mpo
        # create mpo last
        # copy and contract mpo to mps
        scale = self._dimension

        self._sum_north_scaled = self._sum_north * scale

        if self._degeneracy_maps is not None:
            north_degeneracy_map, west_degeneracy_map = self._degeneracy_maps
            tmp_north_deg_num_vals = np.max(north_degeneracy_map) + 1
            tmp_west_deg_num_vals = np.max(west_degeneracy_map) + 1

        influences_mpo = []
        influences_mps = []

        # this block takes care of `i == 0`
        infl = self._influence(0)
        infl = infl / scale
        if self._degeneracy_maps is not None:
            infl_mpo = np.zeros((tmp_west_deg_num_vals, self._dimension**2,
                                 tmp_north_deg_num_vals), \
                                    dtype=np.dtype_complex)
            infl_mps = np.zeros((self._dimension**2,
                                 tmp_north_deg_num_vals), \
                                    dtype=np.dtype_complex)
            # a little bit of optimization is done here by
            # removing the `for` loop and updating slices
            _idxs = np.array(list(range(self._dimension**2)))
            indices = (west_degeneracy_map[_idxs], _idxs,
                       north_degeneracy_map[_idxs])
            influences_mpo.append(np.update(
                array=infl_mpo,
                indices=indices,
                values=infl[indices[2]]
            ))
            influences_mps.append(np.update(
                array=infl_mps,
                indices=indices[1:],
                values=infl[indices[2]] / scale
            ))
        else:
            influences_mpo.append(create_delta(infl, [1, 1, 0]))
            influences_mps.append(infl.T / scale)

        # this block takes care of `i > 0` and `i < self._num_infl - 1`
        # the inner `for` loop can be optimized for parallelization
        if self._num_infl > 2:
            indices = list(range(1, self._num_infl - 1))
            for index in indices:
                infl = self._influence(index)
                influences_mpo.append(create_delta(infl, [0, 1, 1, 0]))
                influences_mps.append(create_delta(infl / scale, [0, 1, 0]))

        # this block takes care of `i == self._num_infl - 1`
        if self._num_infl > 1:
            infl = self._influence(self._num_infl - 1)
            infl_mpo = add_singleton(infl, 1)
            influences_mpo.append(add_singleton(infl_mpo, 3))
            influences_mps.append(add_singleton(infl, 2))

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

        one = np.array([[1.0]], dtype=np.dtype_complex)
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
                if infl is not None:
                    infl_mpo = add_singleton(infl, 1)
                    infl_mpo = add_singleton(infl_mpo, 3)
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
            first_t = add_singleton(first_t, 0)
            tensor = first_t * self._dimension
        elif step == n-1:
            order = [self._mps.bond_edges[-1],self._mps.array_edges[-1][0]]
            last_t = self._mps.nodes[-1].reorder_edges(order).get_tensor()
            last_t = add_singleton(last_t, 1)
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

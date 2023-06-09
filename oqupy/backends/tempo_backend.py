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
Module for tempo and mean-field tempo backend.
"""

from typing import Callable, Dict, List, Optional, Tuple
from copy import copy, deepcopy

from numpy import ndarray, moveaxis, dot, expand_dims, eye
from scipy.linalg import svd as la_svd
from scipy.linalg import block_diag, LinAlgError
from functools import cache

from oqupy import operators
from oqupy.config import TEMPO_BACKEND_CONFIG
from oqupy.backends import node_array as na
from oqupy.util import create_delta


class BaseBackend:
    """
    Backend class for TEMPO.

    Parameters
    ----------
    dimension: int
        Global dimension of the network.
    matrix: callable(int, int) -> ndarray
        Callable that takes the integer coordinates for a node in the 2D network
        and returns the corresponding tensor
    truncation_precision: float
        Maximal relative SVD truncation error.
    max_step: int
        Maximal step the backend computes to.
    max_mps_length: int
        Maximal length the mps can grow to.
    initial_data: ndarray,
        initial array to contract into the initial input leg.
        Must have initial_data.shape = ( n, dimension ) for some n
    """
    def __init__(
            self,
            dimension: int,
            matrix: Callable[[int, int], ndarray],
            truncation_precision: float,
            max_step: Optional[int] = None,
            max_mps_length: Optional[int] = None,
            initial_data: Optional[ndarray] = None,
            config: Optional[Dict] = None):
        """Create a BaseBackend object. """

        self._dimension = dimension
        self._matrix = matrix
        self._precision = truncation_precision

        self._max_step = 500 if max_step is None else max_step  # need oqupy default max?
        self._kmax = max_step if max_mps_length is None else max_mps_length
        self._initial_data = eye(self._dimension) if initial_data is None else initial_data
        self._config = TEMPO_BACKEND_CONFIG if config is None else config

        self._cap = expand_dims(eye(self._dimension), -1)
        self._step = None
        self._mps = None

    @property
    def precision(self) -> float:
        """The svd truncation precision of the TEMPO computation. """
        return self._precision

    @property
    def step(self) -> int:
        """The current step in the TEMPO computation. """
        return self._step

    @property
    def max_step(self) -> int:
        """The maximum step to compute to."""
        return self._max_step

    @max_step.setter
    def max_step(self, n):
        assert n > self._step, 'max step needs to be greater than current step'
        self._max_step = n

    @property
    def kmax(self) -> int:
        """The maximum length of the mps"""
        return self._kmax

    @kmax.setter
    def kmax(self, k):
        assert k > len(self._mps), 'kmax needs to be greater than current step'
        self._kmax = k

    @cache
    def _influence_tensor(self, k):
        """Creates rank 4 influence tensor from rank 2 influence matrix"""
        tensor = self._matrix(self._step, k)
        vert, hor = tensor.shape
        tensor = block_diag(*[v for v in tensor]).reshape((vert * vert, hor))  # block_diag slow?
        tensor = block_diag(*[v for v in tensor.T]).reshape((hor, hor, vert, vert))  # problems when degenerate?
        # should be overall the same as tensor = create_delta(self._matrix(self._step, k), [1, 0, 0, 1])
        return tensor

    def _contract(self, k) -> List:
        """
        Contracts k'th mps site with rank-4 influence tensor
        this contracts mps site with an mpo site to give another mps site with larger bond dims
                                    |
            MPO site        mpo_w --O-- mpo_e
                                    |                                    |
                                             ---->     (mpo_w * mps_w) --O-- (mpo_e * mps_e)
                                    |
            MPS site        mps_w --O-- mps_e

        """
        tens = self._influence_tensor(len(self._mps) - 1 - k)
        if k == 0:
            tens = expand_dims(tens.sum(0), 0)  # sum over mpo_w and create new leg with mpo_w=1

        mps_w, mps_n, mps_e = self._mps[k].shape
        mpo_w, mpo_e, mpo_n, mpo_s = tens.shape

        tmp = tens.dot(self._mps[k])  # new shape (oW, oE, oN, sW, sE)
        tmp = tmp.swapaxes(1, 3)  # new shape (oW, sW, oN, oE, sE)
        tmp = tmp.reshape((mpo_w * mps_w, mpo_n, mpo_e * mps_e))  # new shape (oW * sW, oN, oE * sE)

        self._mps[k] = tmp
        return self._mps[k].shape

    def _truncate_right(self, k) -> int:
        """ Perform svd on kth mps site and truncate the bond with (k+1)'th site"""
        west, north, east = self._mps[k].shape
        theta = self._mps[k].reshape((west * north, east))  # ( -1, self._mps[k].shape[-1])

        u, s, v_dag = self.scipy_svd(theta, self._precision)

        self._mps[k] = u.reshape((west, north, len(s)))  # (self._mps[k].shape[0], -1, len(s) )
        self._mps[k + 1] = u.conj().T.dot(theta).dot(self._mps[k + 1].swapaxes(0, 1))
        return len(s)

    def _truncate_left(self, k) -> int:
        """ Perform svd on kth mps site and truncate the bond with (k-1)'th site"""
        west, north, east = self._mps[k].shape
        theta = self._mps[k].reshape((west, north * east))  # (self._mps[k].shape[0], -1)

        u, s, v_dag = self.scipy_svd(theta, self._precision)

        self._mps[k] = v_dag.reshape((len(s), north, east))  # (len(s), -1, self._mps[k].shape[-1] )
        self._mps[k - 1] = self._mps[k - 1].dot(theta).dot(v_dag.conj().T)
        return len(s)

    def initialise(self) -> List:
        """ ToDo """
        tensor = self._matrix(0, 0).T
        v, h = tensor.shape
        tensor = block_diag(*[row for row in tensor])
        tensor = tensor.reshape((v, v, h))
        tensor = self._initial_data.dot(tensor)
        self._mps = [tensor, self._cap]
        self._step = 1
        return [s.shape for s in self._mps]

    def compute(self) -> List:
        """
        Takes a step in the TEMPO tensor network computation.

        For example, for at step 4, we start with:
        t, current_state_list, current_field)
            A ... self._mps[k]
            B ... self._influence_tensor(step, k)
            c ... self._cap

              |  |  |  |     |
              B~~B~~B~~B~~ ~~c
              |  |  |  |

              |  |  |  |
            ~~A~~A~~A~~A

        Returns
        -------
        bond dimensions: list
            list of the dimensions of mps sites
        """
        if not self._step:
            self.initialise()
        if self._step >= self._max_step:
            print('max step reached')
            return [s.shape for s in self._mps]

        first = 0  # index for the first mps site
        mid = int(len(self._mps) / 2)  # index for the middle mps site
        last = len(self._mps) - 1  # index for the last mps site

        self._contract(first)
        self._contract(last)

        for jj in range(first + 1, mid):  # contract in new tensors, left to middle
            self._contract(jj)
            self._truncate_right(jj - 1)
        for jj in range(last - 1, mid - 1, -1):  # contract in new tensors, right to middle
            self._contract(jj)
            self._truncate_left(jj + 1)
        for jj in range(last, first + 1, -1):  # svd sweep right to left
            self._truncate_left(jj)
        for jj in range(first, last - 1):  # svd sweep left to right: mps now canonical
            self._truncate_right(jj)

        self._mps.append(self._cap)  # turn east leg at last site into north leg at new last site

        if len(self._mps) > self._kmax + 1:
            end = self._mps.pop(0).sum(1)  # remove first site, turn into matrix
            self._mps[0] = end.dot(self._mps[0].swapaxes(0, 1))  # update new first site

        self._step += 1

        return [s.shape for s in self._mps]

    def readout(self) -> ndarray:
        """
                Readout result from mps
                For example, at step 4

                      s  s  s
                      |  |  |  |            |
                    ~~A~~A~~A~~A     ->   ~~R
        """
        result = self._mps[-1].sum(2)
        for m in reversed([s.sum(1) for s in self._mps[:-1]]):
            result = m @ result
        return result

    @staticmethod
    def scipy_svd(theta, precision):
        """ static svd truncation method using scipy.linalg.svd"""
        try:
            u, singular_values, v_dagger = la_svd(theta, full_matrices=False, lapack_driver='gesvd')
        except LinAlgError:
            u, singular_values, v_dagger = la_svd(theta, full_matrices=False, lapack_driver='gesdd')

        try:
            chi = next(i for i in range(len(singular_values))
                       if singular_values[i] / max(singular_values) < precision)
        except StopIteration:
            chi = len(singular_values)
        # chi = len(singular_values)
        return u[:, 0:chi], singular_values[0:chi], v_dagger[0:chi, :]


class BaseTempoBackend:
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
    sum_north: ndarray
        The summing vector for the north legs.
    sum_west: ndarray
        The summing vector for the west legs.
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
            sum_north: ndarray,
            sum_west: ndarray,
            dkmax: int,
            epsrel: float,
            config: Optional[Dict] = None):
        """Create a TempoBackend object. """
        self._initial_state = initial_state
        self._influence = influence
        self._unitary_transform = unitary_transform
        self._sum_north = sum_north
        self._sum_west = sum_west
        self._dkmax = dkmax
        self._epsrel = epsrel
        self._step = None
        self._state = None
        self._config = TEMPO_BACKEND_CONFIG if config is None else config
        self._mps = None
        self._mpo = None
        self._super_u = None
        self._super_u_dagg = None
        self._sum_north_na = None

    @property
    def step(self) -> int:
        """The current step in the TEMPO computation. """
        return self._step

    def initialize_mps_mpo(self) :
        """ToDo"""
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


    def compute_system_step(self, current_step, prop_1, prop_2) -> ndarray:
        """
        Takes a step in the TEMPO tensor network computation.

        For example, for at step 4, we start with:
        t, current_state_list, current_field)
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
                                name="The Time Evolving MPO",
                                copy=False)
        elif current_step <= self._dkmax:
            _, mpo = na.split(self._mpo,
                              int(0 - current_step),
                              copy=True)
        else: # current_step > self._dkmax
            mpo = self._mpo.copy()
            infl = self._influence(self._dkmax-current_step)
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
                            name=f"The MPS ({current_step})")

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
        state = tmp_mps.nodes[0].get_tensor()

        return state


class TempoBackend(BaseTempoBackend):
    """
    ToDo
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
            config: Optional[Dict] = None):
        """Create a TempoBackend object. """
        super().__init__(
            initial_state,
            influence,
            unitary_transform,
            sum_north,
            sum_west,
            dkmax,
            epsrel,
            config)
        self._propagators = propagators

    def initialize(self)-> Tuple[int, ndarray]:
        """
        ToDo
        """
        self._step = 0
        self.initialize_mps_mpo()
        self._state = self._initial_state
        return self._step, copy(self._state)

    def compute_step(self) -> Tuple[int, ndarray]:
        """
        ToDo
        """
        self._step += 1
        prop_1, prop_2 = self._propagators(self._step-1)
        self._state = self.compute_system_step(self._step, prop_1, prop_2)
        return self._step, copy(self._state)


class MeanFieldTempoBackend():
    """
    backend for one or more tensor network tempo with coherent field evolution.
    This creates a list of BaseBackend objects, one for each system in the
    mean-field system, which will be invoked at each timesteps to propagate the
    systems concurrently. In addition, at each timestep a coherent field is
    evolved according to the state of each system and field value at that time.

    Parameters
    ----------
    initial_state_list: List[ndarray]
        List of initial density matrices (as vectors), one for each system.
    initial_field: complex
        The initial field value.
    influence_list: List[callable(int) -> ndarray]
        Callables that takes an integer `step` and returns the influence super
        operator of that `step`, one for each system.
    unitary_transform_list: List[ndarray]
        Unitaries transforms the coupling operator into a diagonal form, one
        for each system (i.e. the bath associated with each system).
    propagators_list: List[callable(int, complex, complex) -> ndarray, ndarray]
        Callables that takes an integer `step`, a complex `field` and a complex
        `field_derivative` and returns the first and second half of the system
        propagator of that `step`. One for each system.
    compute_field: callable(int, List[ndarray], complex,
                            Optional[List[ndarray]]) -> complex
        Callable that takes an integer `step`, a complex `field` (the current
        value of the field) and two lists of ndarrays for, respectively, the
        current and next density matrix of each system, and returns the next
        field value.
    compute_field_derivative: callable(int, List[ndarray], complex) -> complex
        Callable that takes an integer `step`, a complex `field` (the current
        value of the field) and a list of vectors for the density matrix of each
        system at `step`, and returns the field derivative at `step`.
    sum_north_list: List[ndarray]
        The summing vector for the north legs of each system's tensor network.
    sum_west_list: List[ndarray]
        The summing vector for the west legs of each system's tensor network.
    dkmax: int
        Number of influences to include. If ``dkmax == -1`` then all influences
        are included. Applies to all systems.
    epsrel: float
        Maximal relative SVD truncation error. Applies to all systems.
    """
    def __init__(
            self,
            initial_state_list: List[ndarray],
            initial_field: complex,
            influence_list: List[Callable[[int], ndarray]],
            unitary_transform_list: List[ndarray],
            propagators_list: List[Callable[[int, complex, complex],
                Tuple[ndarray, ndarray]]],
            compute_field: Callable[[float, List[ndarray], complex,
                                     Optional[List[ndarray]]], complex],
            compute_field_derivative:
                Callable[[float, List[ndarray], complex], complex],
            sum_north_list: List[ndarray],
            sum_west_list: List[ndarray],
            dkmax: int,
            epsrel: float,
            config: Dict):
        """Create a MeanFieldTempoBackend object. """
        self._initial_state_list = initial_state_list
        self._initial_field = initial_field
        self._compute_field = compute_field
        self._compute_field_derivative = compute_field_derivative
        self._field = initial_field
        self._state_list = initial_state_list
        self._step = None
        self._propagators_list = propagators_list
        # List of BaseTempoBackends use to calculate each system dynamics
        self._backend_list = [BaseTempoBackend(initial_state,
                         influence,
                         unitary_transform,
                         sum_north,
                         sum_west,
                         dkmax,
                         epsrel,
                         config)
                         for initial_state, influence, unitary_transform,
                         sum_north, sum_west in zip(initial_state_list,
                             influence_list, unitary_transform_list,
                             sum_north_list, sum_west_list)]

    @property
    def step(self) -> int:
        """The current step in the TEMPO computation. """
        return self._step

    def initialize(self) -> Tuple[int, ndarray, complex]:
        """Initialize each TEMPO instance. """
        self._step = 0
        for backend in self._backend_list:
            backend.initialize_mps_mpo()
        return self._step, deepcopy(self._state_list), self._field

    def compute_step(self) -> Tuple[int, List[ndarray], complex]:
        """Calculate the next step of the MeanFIeldSystem
        dynamics"""
        current_step = self._step
        next_step = current_step + 1
        current_state_list = deepcopy(self._state_list)
        current_field = self._field
        current_field_derivative = self._compute_field_derivative(
            current_step, current_state_list, current_field)
        # N.B. propagators use current_field & current_field_derivative
        # this is how field dependence enters in each system dynamics
        prop_tuple_list = [propagators(current_step, current_field,
            current_field_derivative) for propagators, state
            in zip(self._propagators_list, current_state_list)]
        # Use tempo tensor network to compute each system state
        next_state_list = [backend.compute_system_step(next_step,
            *prop_tuple) for backend, prop_tuple
            in zip(self._backend_list, prop_tuple_list)]
        # Use field evolution function to compute next field
        next_field = self._compute_field(current_step,
            current_state_list, current_field,
            next_state_list)
        self._state_list = next_state_list
        self._field = next_field
        self._step = next_step

        return self._step, deepcopy(self._state_list), self._field

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

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np

from numpy import ndarray, moveaxis, dot, expand_dims, eye, exp, outer,\
    diag, array, kron, ones, ascontiguousarray, swapaxes, reshape, amax, argmax
from scipy.linalg import svd
from scipy.linalg import block_diag, LinAlgError
from functools import cache
from numpy.linalg import svd as nsvd
from oqupy import operators
from oqupy.config import TEMPO_BACKEND_CONFIG
from oqupy.backends import node_array as na
from oqupy.util import create_delta
import numpy as np
from numba import jit
# import jax.numpy as jnp

# def signif(x, p):
#     x = np.asarray(x)
#     x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
#     mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
#     return np.round(x * mags) / mags
# def _scipy_svd(theta, precision):
#     """ static svd truncation method using scipy.linalg.svd"""
#     try:
#         u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesvd')
#     except LinAlgError:
#         u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesdd')
#     #u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesvd')
#     #u, singular_values, v_dagger = np.linalg.svd(theta)
#     chi = argmax(singular_values/amax(singular_values) < precision)
#     if not chi:
#         chi = len(singular_values)
#
#     return u[:, 0:chi], singular_values[0:chi], v_dagger[0:chi, :]

# @jit(nopython=True)
# def _numpy_svd(theta, precision):
#     u, s, vd = np.linalg.svd(theta)
#     chi = np.argmax(s / np.amax(s) < precision)
#     if not chi:
#         chi = len(s)
#     return u[:, 0:chi], s[0:chi], vd[0:chi, :]
#
# def numpy_svd(theta, precision):
#     u, s, vd = np.linalg.svd(theta)
#     chi = np.argmax(s / np.amax(s) < precision)
#     if not chi:
#         chi = len(s)
#     return u[:, 0:chi], s[0:chi], vd[0:chi, :]
# from time import time
# mat = np.random.rand(1000, 1000)
# t0 = time()
# r1 = _numpy_svd(mat, 10**(-9))
# print(time()-t0)
# t0 = time()
# r1 = _numpy_svd(mat, 10**(-9))
# print(time()-t0)
# t0 = time()
# r1 = numpy_svd(mat, 10**(-9))
# print(time()-t0)
# t0 = time()
# r1 = _scipy_svd(mat, 10**(-9))
# print(time()-t0)
# exit()
class TIBaseBackend:
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
            truncation_precision: float,
            # propagators: Callable[[int], List[ndarray]],
            propagator: ndarray,
            coefficients: Callable[[int], complex],
            operators: Tuple[ndarray],
            max_step: Optional[int] = None,
            max_mps_length: Optional[int] = None,
            initial_data: Optional[ndarray] = None,
            config: Optional[Dict] = None):
        """Create a BaseBackend object. """

        self._dim = dimension
        self._precision = truncation_precision
        # self._propagators = propagators
        self._coefficients = coefficients
        self._ops = operators
        self._prop = propagator

        self._max_step = max_step if max_step is None else max_step  # need oqupy default max?
        self._kmax = max_step if max_mps_length is None else max_mps_length
        self._initial_data = eye(self._dim) if initial_data is None else initial_data
        self._config = TEMPO_BACKEND_CONFIG if config is None else config

        self._h_ind, self._h_proj = self._unique(self._ops[0])
        self._v_ind, self._v_proj = self._unique(zip(*self._ops[1:]))
        self._v_dim, self._h_dim, = len(self._v_ind), len(self._h_ind)

        # self._v_dim, self._v_ind, self._v_proj = self._dim, list(range(self._dim)), eye(self._dim)
        # self._h_dim, self._h_ind, self._h_proj = self._dim, list(range(self._dim)), eye(self._dim)

        self._cap = expand_dims(eye(self._dim), -1)
        self._step = None
        self._mps = None
        self.data = [self._initial_data]

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
        c = self._coefficients(k + 1)
        o_1 = self._ops[0]
        o_2 = c.real * self._ops[1] - 1j * c.imag * self._ops[2]

        prop = self._prop
        if k == 0:
            c0 = self._coefficients(0)
            o0_2 = c0.real * self._ops[1] - 1j * c0.imag * self._ops[2]
            tensor = exp(outer(o_2, o_1)) * dot(prop, prop).T * exp(o_1 * o0_2)
            tensor = dot(kron(self._v_proj, self._h_proj), diag(tensor.flatten()))
            tensor = reshape(tensor, (self._v_dim, self._h_dim, self._dim, self._dim))
            tensor = tensor # * dot(prop, prop).T * exp(o_1 * o0_2)
            tensor = moveaxis(swapaxes(tensor, 2, 3), 0, 2)
        else:
            tensor = diag(exp(kron(o_2[self._v_ind], o_1[self._h_ind])))
            tensor = reshape(tensor, (self._v_dim, self._h_dim, self._v_dim, self._h_dim))
            tensor = swapaxes(tensor, 0, 3)
        return tensor
    # @cache
    # def _influence_matrix(self, k):
    #     c = self._coefficients(k + 1)
    #     o_1 = self._ops[0]
    #     o_2 = c.real * self._ops[1] - 1j * c.imag * self._ops[2]
    #     if k == 0:
    #         c0 = self._coefficients(0)
    #         o0_2 = c0.real * self._ops[1] - 1j * c0.imag * self._ops[2]
    #         return exp(outer(o_2, o_1)) * dot(self._prop, self._prop) * exp(o_1 * o0_2)
    #     else:
    #         return exp(outer(o_2[self._v_ind], o_1[self._h_ind]))

    # def _contract2(self, k) -> List:
    #     """
    #     Contracts k'th mps site with rank-4 influence tensor
    #     this contracts mps site with an mpo site to give another mps site with larger bond dims
    #                                 |
    #         MPO site        mpo_w --O-- mpo_e
    #                                 |                                    |
    #                                          ---->     (mpo_w * mps_w) --O-- (mpo_e * mps_e)
    #                                 |
    #         MPS site        mps_w --O-- mps_e
    #
    #     """
    #     mat = self._influence_matrix(len(self._mps) - 1 - k)
    #     v, h = mat.shape
    #     w, n, e = self._mps[k].shape
    #
    #     if k == 0:
    #         self._mps[k] = array([v * swapaxes(self._mps[k], 1, 2) for v in mat.T]) # shape (h, w, e, n)
    #         self._mps[k] = swapaxes(self._mps[k], 0, 1)  # shape (w, h, e, n)
    #         self._mps[k] = reshape(self._mps[k], (w, h * e, n))
    #         self._mps[k] = swapaxes(self._mps[k], 1, 2)
    #         return self._mps[k].shape
    #
    #     self._mps[k] = reshape(swapaxes(self._mps[k], 1, 2), (-1, n)) # shape (w * e, n)
    #     self._mps[k] = block_diag(*[v * self._mps[k] for v in mat.T])  # shape (h * w * e, h * n)
    #     if k == len(self._mps) - 1:
    #         self._mps[k] = reshape(self._mps[k], (h, h * w * e * n))
    #         self._mps[k] = dot(self._h_proj, self._mps[k]) # shape (nh, h * w * e * n)
    #         self._mps[k] = dot(reshape(self._mps[k], (self._h_dim * h * w * e, n)), self._v_proj.T)
    #         # shape (nh * h * w * e, nn)
    #         self._mps[k] = reshape(self._mps[k], (self._h_dim, h, w, e, self._v_dim))
    #         self._mps[k] = reshape(swapaxes(self._mps[k], 1, 2), (self._h_dim * w, h * e, self._v_dim))
    #         return self._mps[k].shape
    #
    #     self._mps[k] = reshape(self._mps[k], (h, h, w, e, n))
    #     self._mps[k] = swapaxes(reshape(swapaxes(self._mps[k], 1, 2), (h * w, h * e, n)), 1, 2)
    #     return self._mps[k].shape

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
        # self._mps[k] = block_diag(*[v * swapaxes(self._mps[k] , 1, 2) for v in mat.T])
        tens = self._influence_tensor(len(self._mps) - 1 - k)
        if k == 0:
            tens = expand_dims(tens.sum(0), 0)  # sum over mpo_w and create new leg with mpo_w=1

        mps_w, mps_n, mps_e = self._mps[k].shape
        mpo_w, mpo_e, mpo_n, mpo_s = tens.shape

        tmp = dot(tens, self._mps[k])  # new shape (oW, oE, oN, sW, sE)
        tmp = tmp.swapaxes(1, 3)  # new shape (oW, sW, oN, oE, sE)
        tmp = tmp.reshape((mpo_w * mps_w, mpo_n, mpo_e * mps_e))  # new shape (oW * sW, oN, oE * sE)

        self._mps[k] = tmp
        return self._mps[k].shape

    def _truncate_right(self, k) -> int:
        """ Perform svd on kth mps site and truncate the bond with (k+1)'th site"""
        west, north, east = self._mps[k].shape
        u, s, v_dag = self._scipy_svd(reshape(self._mps[k], (west * north, east)), self._precision)
        self._mps[k] = reshape(u, (west, north, len(s)))  # (self._mps[k].shape[0], -1, len(s) )
        self._mps[k + 1] = dot((s * v_dag.T).T, swapaxes(self._mps[k + 1], 0, 1))
        return len(s)

    def _truncate_left(self, k) -> int:
        """ Perform svd on kth mps site and truncate the bond with (k+1)'th site"""
        self._mps[k] = self._mps[k].T
        east, north, west = self._mps[k].shape
        u, s, v_dag = self._scipy_svd(reshape(self._mps[k], (east * north, west)), self._precision)
        self._mps[k] = reshape(u, (east, north, len(s))).T  # (self._mps[k].shape[0], -1, len(s) )
        self._mps[k - 1] = dot(self._mps[k - 1], v_dag.T * s)
        return len(s)

    # def _truncate_right2(self, k) -> int:
    #     """ Perform svd on kth mps site and truncate the bond with (k+1)'th site"""
    #
    #     west, north, east = self._mps[k].shape
    #     self._mps[k] = reshape(self._mps[k], (self._h_dim, int(west/self._h_dim) * north,
    #                                           int(east/self._h_dim), self._h_dim))
    #     svd_results = [self._scipy_svd(self._mps[k][i, :, :, i], self._precision) for i in range(self._h_dim)]
    #
    #     u, s, v_dag = _numpy_svd(reshape(self._mps[k], (west * north, east)), self._precision)
    #     self._mps[k] = reshape(u, (west, north, len(s)))  # (self._mps[k].shape[0], -1, len(s) )
    #     self._mps[k + 1] = dot((s * v_dag.T).T, swapaxes(self._mps[k + 1], 0, 1))
    #     return len(s)
    #
    # def _truncate_left_slow(self, k) -> int:
    #     """ Perform svd on kth mps site and truncate the bond with (k-1)'th site"""
    #     west, north, east = self._mps[k].shape
    #     u, s, v_dag = _numpy_svd(self._mps[k].reshape((west, north * east)), self._precision)
    #     self._mps[k] = reshape(v_dag, (len(s), north, east))  # (len(s), -1, self._mps[k].shape[-1] )
    #     self._mps[k - 1] = dot(self._mps[k - 1], u * s)
    #
    #     # westm, northm, eastm = self._mps[k-1].shape
    #     # self._mps[k - 1] = reshape(dot(reshape(self._mps[k - 1], (-1, eastm)), u * s),
    #     #                            (westm, northm, len(s)))
    #     return len(s)

    def initialise(self, step=None, mps=None) -> Tuple[int, ndarray]:
        """ ToDo """
        if mps is not None:
            self._mps = mps
            self._step = step
        else:
            c_real, c_imag = self._coefficients(0).real, self._coefficients(0).imag
            o_1 = self._ops[0]
            o_2 = c_real * self._ops[1] - 1j * c_imag * self._ops[2]
            tensor = dot(self._initial_data, self._prop.T * exp(o_1 * o_2))  # shape ( e, n, s)
            self.data.append(dot(tensor, self._prop.T))
            tensor = dot(self._influence_tensor(0), tensor.T) # contains whole timestep freeprop!
            tensor = swapaxes(tensor.sum(0), 0, 2)
            self._mps = [tensor, self._cap]
            self.data.append(self.readout())
            self._step = 1
        return self._step, self.data[-1]

    def compute_step(self) -> Tuple[int, ndarray]:  ## make readout optional
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
        # if not self._step:
        #     self.initialise()
        # if self._step >= self._max_step + 2:
        #     print('max step reached')
        #     return [s.shape for s in self._mps]

        first = 0  # index for the first mps site
        mid = int(len(self._mps) / 2)   # index for the middle mps site
        last = len(self._mps) - 1  # index for the last mps site

        self._contract(first)
        self._contract(last)

        for jj in range(last - 1, mid - 1, -1):  # contract in new tensors, right to middle
            self._contract(jj)
            self._truncate_left(jj + 1)
        for jj in range(first + 1, mid):  # contract in new tensors, left to middle
            self._contract(jj)
            self._truncate_right(jj - 1)
        for jj in range(first, last):  # svd sweep left to right: mps now canonical
            self._truncate_right(jj)
        for jj in range(last, first, -1):  # svd sweep right to left
            self._truncate_left(jj)

        # for jj in range(first + 1, mid):  # contract in new tensors, left to middle
        #     self._contract(jj)
        #     self._truncate_right(jj - 1)
        # for jj in range(last - 1, mid - 1, -1):  # contract in new tensors, right to middle
        #     self._contract(jj)
        #     self._truncate_left(jj + 1)
        # for jj in range(last - 1, first, -1):  # svd sweep right to left
        #     self._truncate_left(jj + 1)
        # for jj in range(first, last):  # svd sweep left to right: mps now canonical
        #     self._truncate_right(jj)
        #
        # for jj in range(last, first, -1):  # svd sweep left to right: mps now canonical
        #     self._truncate_left(jj)
        # for jj in range(first, last):  # svd sweep left to right: mps now canonical
        #     self._truncate_right(jj)

        self._mps.append(self._cap)  # turn east leg at last site into north leg at new last site

        if len(self._mps) > self._kmax + 1:  ## check this
            end = self._mps.pop(0).sum(1)  # remove first site, turn into matrix
            self._mps[0] = dot(end, swapaxes(self._mps[0], 0, 1))  # dot into new first site

        self._step += 1
        self.data.append(self.readout())

        return self._step, self.data[-1]

    def readout(self) -> ndarray:
        """
                Readout result from mps
                For example, at step 4

                      s  s  s
                      |  |  |  |            |
                    ~~A~~A~~A~~A     ->   ~~R
        """
        result = self._prop.T
        for m in reversed([s.sum(1) for s in self._mps[:-1]]):
            result = m @ result
        return result

    @staticmethod
    def _scipy_svd(theta, precision):
        """ static svd truncation method using scipy.linalg.svd"""
        try:
            u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesvd')
        except LinAlgError:
            print('svd except')
            u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesdd')
        chi = argmax(singular_values/amax(singular_values) < precision)
        if not chi:
            chi = len(singular_values)
        return u[:, 0:chi], singular_values[0:chi], v_dagger[0:chi, :]

    @staticmethod
    def _unique(values):
        vals = list(values)
        inverse = [vals.index(e) for e in vals]
        indices = array(sorted(set(inverse), key=inverse.index))
        inverse = array([[int(i == j) for i in inverse] for j in indices])
        return indices, inverse
#
# class BaseBackend:
#     """
#     Backend class for TEMPO.
#
#     Parameters
#     ----------
#     dimension: int
#         Global dimension of the network.
#     matrix: callable(int, int) -> ndarray
#         Callable that takes the integer coordinates for a node in the 2D network
#         and returns the corresponding tensor
#     truncation_precision: float
#         Maximal relative SVD truncation error.
#     max_step: int
#         Maximal step the backend computes to.
#     max_mps_length: int
#         Maximal length the mps can grow to.
#     initial_data: ndarray,
#         initial array to contract into the initial input leg.
#         Must have initial_data.shape = ( n, dimension ) for some n
#     """
#
#     def __init__(
#             self,
#             dimension: int,
#             truncation_precision: float,
#             propagators: Callable[[int], List[ndarray]],
#             coefficients: Callable[[int], complex],
#             operators: Tuple[ndarray],
#             max_step: Optional[int] = None,
#             max_mps_length: Optional[int] = None,
#             initial_data: Optional[ndarray] = None,
#             config: Optional[Dict] = None):
#         """Create a BaseBackend object. """
#
#         self._dim = dimension
#         self._precision = truncation_precision
#         self._propagators = propagators
#         self._coefficients = coefficients
#         self._ops = operators
#
#         self._max_step = max_step if max_step is None else max_step  # need oqupy default max?
#         self._kmax = max_step if max_mps_length is None else max_mps_length
#         self._initial_data = eye(self._dim) if initial_data is None else initial_data
#         self._config = TEMPO_BACKEND_CONFIG if config is None else config
#
#         self._h_ind, self._h_proj = self._unique(self._ops[0])
#         self._v_ind, self._v_proj = self._unique(zip(*self._ops[1:]))
#         self._v_dim, self._h_dim, = len(self._v_ind), len(self._h_ind)
#
#         self._cap = expand_dims(eye(self._dim), -1)
#         self._step = None
#         self._mps = None
#
#     @property
#     def precision(self) -> float:
#         """The svd truncation precision of the TEMPO computation. """
#         return self._precision
#
#     @property
#     def step(self) -> int:
#         """The current step in the TEMPO computation. """
#         return self._step
#
#     @property
#     def max_step(self) -> int:
#         """The maximum step to compute to."""
#         return self._max_step
#
#     @max_step.setter
#     def max_step(self, n):
#         assert n > self._step, 'max step needs to be greater than current step'
#         self._max_step = n
#
#     @property
#     def kmax(self) -> int:
#         """The maximum length of the mps"""
#         return self._kmax
#
#     @kmax.setter
#     def kmax(self, k):
#         assert k > len(self._mps), 'kmax needs to be greater than current step'
#         self._kmax = k
#
#     @cache
#     def _influence_tensor(self, k):
#         """Creates rank 4 influence tensor from rank 2 influence matrix"""
#         c_real, c_imag = self._coefficients(k).real, self._coefficients(k).imag
#         o_1 = self._ops[0]
#         o_2 = c_real * self._ops[1] - 1j * c_imag * self._ops[2]
#
#         if k == 0:
#             tensor = diag(diag(exp(o_1 * o_2)).flatten())
#             tensor = kron(self._v_proj, self._h_proj).dot(tensor)
#             tensor = reshape(tensor, (self._v_dim, self._h_dim, self._dim, self._dim))
#
#             # tensor = kron(eye(self._dim), self._h_proj).dot(tensor)
#             # tensor = reshape(tensor, (self._dim, self._h_dim, self._dim, self._dim))
#             tensor = moveaxis(tensor, 0, 2)
#         else:
#             tensor = diag(exp(kron(o_2[self._v_ind], o_1[self._h_ind])))
#             tensor = reshape(tensor, (self._v_dim, self._h_dim, self._v_dim, self._h_dim))
#             tensor = swapaxes(tensor, 0, 3)
#         return tensor
#
#     def _tensor(self, n, k):
#         tensor = self._influence_tensor(k)
#         if k == 0:
#             prop_1, prop_2 = self._propagators(n)
#             tensor = dot(tensor, prop_1)  # transpose prop?
#             tensor = swapaxes(dot(prop_2, swapaxes(tensor, 1, 2)), 0, 1)
#         return tensor
#
#     def _contract(self, k) -> List:
#         """
#         Contracts k'th mps site with rank-4 influence tensor
#         this contracts mps site with an mpo site to give another mps site with larger bond dims
#                                     |
#             MPO site        mpo_w --O-- mpo_e
#                                     |                                    |
#                                              ---->     (mpo_w * mps_w) --O-- (mpo_e * mps_e)
#                                     |
#             MPS site        mps_w --O-- mps_e
#
#         """
#         tens = self._tensor(self._step, len(self._mps) - 1 - k)
#         if k == 0:
#             tens = expand_dims(tens.sum(0), 0)  # sum over mpo_w and create new leg with mpo_w=1
#         # elif k == len(self._mps) - 1:
#         #     tens = expand_dims(tens.sum(1), 1)
#
#         mps_w, mps_n, mps_e = self._mps[k].shape
#         mpo_w, mpo_e, mpo_n, mpo_s = tens.shape
#
#         tmp = tens.dot(self._mps[k])  # new shape (oW, oE, oN, sW, sE)
#         tmp = tmp.swapaxes(1, 3)  # new shape (oW, sW, oN, oE, sE)
#         tmp = tmp.reshape((mpo_w * mps_w, mpo_n, mpo_e * mps_e))  # new shape (oW * sW, oN, oE * sE)
#
#         self._mps[k] = tmp
#         return self._mps[k].shape
#
#     def _truncate_right(self, k) -> int:
#         """ Perform svd on kth mps site and truncate the bond with (k+1)'th site"""
#         west, north, east = self._mps[k].shape
#
#         # theta = self._mps[k].reshape((west * north, east))  # ( -1, self._mps[k].shape[-1])
#         # u, s, v_dag = self._scipy_svd(theta, self._precision)
#         # self._mps[k] = u.reshape((west, north, len(s)))  # (self._mps[k].shape[0], -1, len(s) )
#         # self._mps[k + 1] = u.conj().T.dot(theta).dot(self._mps[k + 1].swapaxes(0, 1))
#
#         u, s, v_dag = self._scipy_svd(reshape(self._mps[k], (west * north, east)), self._precision)
#
#         # indi, proj = self._unique(s)
#         # print(len(indi), len(s))
#         #
#         # self._mps[k] = u.dot(proj.T).reshape((west, north, len(indi)))  # (self._mps[k].shape[0], -1, len(s) )
#         # self._mps[k + 1] = proj.dot(diag(s)).dot(v_dag).dot(self._mps[k + 1].swapaxes(0, 1))
#
#         self._mps[k] = reshape(u, (west, north, len(s)))  # (self._mps[k].shape[0], -1, len(s) )
#         #self._mps[k + 1] = diag(s).dot(v_dag).dot(self._mps[k + 1].swapaxes(0, 1))
#         self._mps[k + 1] = dot((s * v_dag.T).T, swapaxes(self._mps[k + 1], 0, 1))
#         return len(s)
#
#     def _truncate_left(self, k) -> int:
#         """ Perform svd on kth mps site and truncate the bond with (k-1)'th site"""
#         west, north, east = self._mps[k].shape
#
#         # theta = self._mps[k].reshape((west, north * east))  # (self._mps[k].shape[0], -1)
#         # u, s, v_dag = self._scipy_svd(theta, self._precision)
#
#         # self._mps[k] = v_dag.reshape((len(s), north, east))  # (len(s), -1, self._mps[k].shape[-1] )
#         # self._mps[k - 1] = self._mps[k - 1].dot(theta).dot(v_dag.conj().T)
#
#         #theta = self._mps[k].reshape((west, north * east))  # (self._mps[k].shape[0], -1)
#         u, s, v_dag = self._scipy_svd(self._mps[k].reshape((west, north * east)), self._precision)
#
#         self._mps[k] = reshape(v_dag, (len(s), north, east))  # (len(s), -1, self._mps[k].shape[-1] )
#         #self._mps[k - 1] = self._mps[k - 1].dot(u).dot(diag(s))
#         self._mps[k - 1] = dot(self._mps[k - 1], u * s)
#         return len(s)
#
#     def initialise(self, step=None, mps=None) -> List:
#         if mps is not None:
#             self._mps = mps
#             self._step = step
#         else:
#             tensor = self._tensor(0, 0).sum(0)  # shape (e, n, s)
#             tensor = tensor.dot(self._initial_data.T).swapaxes(0, 2)
#             self._mps = [tensor, self._cap]
#             self._step = 1
#         return [s.shape for s in self._mps]
#
#     def compute(self) -> List:
#         """
#         Takes a step in the TEMPO tensor network computation.
#
#         For example, for at step 4, we start with:
#         t, current_state_list, current_field)
#             A ... self._mps[k]
#             B ... self._influence_tensor(step, k)
#             c ... self._cap
#
#               |  |  |  |     |
#               B~~B~~B~~B~~ ~~c
#               |  |  |  |
#
#               |  |  |  |
#             ~~A~~A~~A~~A
#
#         Returns
#         -------
#         bond dimensions: list
#             list of the dimensions of mps sites
#         """
#         if not self._step:
#             self.initialise()
#         if self._step >= self._max_step + 2:
#             print('max step reached')
#             return [s.shape for s in self._mps]
#
#         first = 0  # index for the first mps site
#         mid = int(len(self._mps) / 2)  # index for the middle mps site
#         last = len(self._mps) - 1  # index for the last mps site
#
#         self._contract(first)
#         self._contract(last)
#
#         for jj in range(first + 1, mid):  # contract in new tensors, left to middle
#             self._contract(jj)
#             self._truncate_right(jj - 1)
#         for jj in range(last - 1, mid - 1, -1):  # contract in new tensors, right to middle
#             self._contract(jj)
#             self._truncate_left(jj + 1)
#         for jj in range(mid - 1, first, -1):  # svd sweep right to left
#             self._truncate_left(jj + 1)
#         for jj in range(first, last):  # svd sweep left to right: mps now canonical
#             self._truncate_right(jj)
#
#         # for jj in range(last - 1, mid - 1, -1):  # contract in new tensors, right to middle
#         #     self._contract(jj)
#         #     self._truncate_left(jj + 1)
#         # for jj in range(first + 1, mid):  # contract in new tensors, left to middle
#         #     self._contract(jj)
#         #     self._truncate_right(jj - 1)
#         # for jj in range(first, last - 1):  # svd sweep left to right: mps now canonical
#         #     self._truncate_right(jj)
#         # for jj in range(last, first, -1):  # svd sweep right to left
#         #     self._truncate_left(jj)
#
#         self._mps.append(self._cap)  # turn east leg at last site into north leg at new last site
#
#         if len(self._mps) > self._kmax + 2:  ## check this
#             end = self._mps.pop(0).sum(1)  # remove first site, turn into matrix
#             self._mps[0] = dot(end, swapaxes(self._mps[0], 0, 1))  # dot into new first site
#
#         self._step += 1
#
#         return [s.shape for s in self._mps]
#
#     def readout(self) -> ndarray:
#         """
#                 Readout result from mps
#                 For example, at step 4
#
#                       s  s  s
#                       |  |  |  |            |
#                     ~~A~~A~~A~~A     ->   ~~R
#         """
#         result = self._mps[-1].sum(2)
#         for m in reversed([s.sum(1) for s in self._mps[:-1]]):
#             result = m @ result
#         return result
#
#     @staticmethod
#     def _scipy_svd(theta, precision):
#         """ static svd truncation method using scipy.linalg.svd"""
#         try:
#             u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesvd')
#         except LinAlgError:
#             u, singular_values, v_dagger = svd(theta, full_matrices=False, lapack_driver='gesdd')
#         # try:
#         #     chi = next(i for i in range(len(singular_values))
#         #                if singular_values[i] / max(singular_values) < precision)
#         # except StopIteration:
#         #     chi = len(singular_values)
#
#         chi = argmax(singular_values / amax(singular_values) < precision)
#         if not chi:
#             chi = len(singular_values)
#         # vals = list(singular_values)
#         # inverse = [vals.index(e) for e in vals]
#         # indices = array(sorted(set(inverse), key=inverse.index))
#         # print(len(singular_values), len(indices))
#         # print(singular_values)
#         return u[:, 0:chi], singular_values[0:chi], v_dagger[0:chi, :]
#
#     @staticmethod
#     def _unique(values):
#         vals = list(values)
#         inverse = [vals.index(e) for e in vals]
#         indices = array(sorted(set(inverse), key=inverse.index))
#         inverse = array([[int(i == j) for i in inverse] for j in indices])
#         return indices, inverse

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

    def initialize_mps_mpo(self):
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
        else:  # current_step > self._dkmax
            mpo = self._mpo.copy()
            infl = self._influence(self._dkmax - current_step)
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
                         axes=[(0, 0)],
                         left_index=-1,
                         right_index=-1,
                         direction="left",
                         max_singular_values=None,
                         max_truncation_err=self._epsrel,
                         relative=True,
                         copy=False)

        if len(self._mps) != len(mpo):
            self._mps.contract(self._sum_north_na,
                               axes=[(0, 0)],
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
        for _ in range(len(tmp_mps) - 1):
            tmp_mps.contract(self._sum_north_na,
                             axes=[(0, 0)],
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

    def initialize(self) -> Tuple[int, ndarray]:
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
        prop_1, prop_2 = self._propagators(self._step - 1)
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

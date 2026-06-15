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
Module for the time translational invariant time evolving matrix product
operator algorithm (TTI-TEMPO) backend. This module is based on [Link2024].

**[Link2024]**
V. Link, H. Tu, and W. T. Strunz, *Open Quantum System Dynamics from Infinite
Tensor Network Contraction*, `Phys. Rev. Lett. 132, 200403
<https://doi.org/10.1103/PhysRevLett.132.200403>`__ (2024).

Original code taken from https://github.com/val-link/iTEBD-TEMPO.git
# Implementation of iTEBD-TEMPO
# Author: Valentin Link (valentin.link@tu-dresden.de)
# Original code from https://github.com/val-link/iTEBD-TEMPO.git
# Modified by Paul Eastham (easthamp@tcd.ie) to use the OQuPy BathCorrelations
# class to define the bath correlations rather than the bath correlation
# function itself. 
# Please cite the corresponding publication:
# https://doi.org/10.1103/PhysRevLett.132.200403.
"""

from typing import Callable, Dict

import numpy as np
from scipy.linalg import norm, svd
from scipy.sparse.linalg import eigs

from oqupy.process_tensor import SimpleProcessTensorInfinite

class TTITempoBackend():
    """
    Base class for TTI process tensor tempo backends.

    Parameters
    ----------
    influence: callable(int) -> ndarray
        Callable that takes an integer `step` and returns the influence super
        operator of that `step`.
    process_tensor: BaseProcessTensor
        Todo
    dkmax: int
        Number of influences to include. If ``dkmax == None`` then all
        influences are included.
    epsrel: float
        Maximal relative SVD truncation error.
    rank: int
        Maximal SVD truncation rank
    config: Dict
        Todo
    """

    def __init__(
            self,
            dimension: int,
            influence: Callable[[int], np.ndarray],
            process_tensor: SimpleProcessTensorInfinite,
            dkmax: int,
            epsrel: float,
            rank: int,
            config: Dict):

        self.repeating_cell = (1, 1)
        self._dkmax = dkmax
        self._dimension = dimension
        self._influence = influence
        self._process_tensor = process_tensor
        self._nu_dim = self._dimension**2 + 1
        self._epsrel = epsrel
        self._rank = rank
        self._itol = config["itol"]
        self._A = None
        self._B = None
        self._sAB = None
        self._sBA = None
        self._rank_is_one = None
        self._step = None

    @property
    def step(self) -> int:
        """The current step in the TTI-TEMPO computation. """
        return self._step

    @property
    def num_steps(self) -> int:
        """The current step in the TTI-TEMPO computation. """
        return self._dkmax

    def initialize(self) -> None:
        """Initializes the iTEBD algorithm. """
        self._A = np.ones((1, self._nu_dim, 1))
        self._B = np.ones((1, self._nu_dim, 1))
        self._sAB = np.ones((1))
        self._sBA = np.ones((1))
        self._rank_is_one = True
        self._step = 1

    def compute_step(self) -> None:
        """
        Compute a step of the iTEBD algorithm.
        """
        i_tens = self._influence(self._dkmax - self._step)

        if self._step % 2 == 0:
            self._sBA, self._sAB = self._sAB, self._sBA
            self._B, self._A, = self._A, self._B

        # renormalize weights
        self._sAB = self._sAB * norm(self._sBA)
        self._sBA = self._sBA / norm(self._sBA)

        # ensure weights are above tolerance (needed for inversion)
        self._sBA[np.abs(self._sBA) < self._itol] = self._itol

        # MPS - gate contraction
        d1 = i_tens.shape[1]
        d2 = i_tens.shape[-1]
        rank_BA = self._sBA.shape[0]

        if self._step == self._dkmax:
            d1 = 1
            u, s_vals, v = svd(np.einsum('a,acd,d,dcg,g,i,c->aicg', self._sBA,
                                         self._A, self._sAB, self._B, self._sBA,
                                         np.ones((1)), np.diagonal(i_tens)
                                         ).reshape([d1*rank_BA, d2*rank_BA]),
                               full_matrices=False)
        else:
            u, s_vals, v = svd(np.einsum('a,acd,d,dfg,g,fc->afcg', self._sBA,
                                         self._A, self._sAB, self._B, self._sBA,
                                         i_tens
                                         ).reshape([d1*rank_BA, d2*rank_BA]),
                               full_matrices=False)

        # truncate singular values
        if self._epsrel is None:
            rank_new = min(self._rank, len(s_vals))
        else:
            s_vals_sum = np.cumsum(s_vals) / np.sum(s_vals)
            rank_rtol = np.searchsorted(s_vals_sum, 1 - self._epsrel) + 1
            rank_new = min(self._rank, len(s_vals), rank_rtol)
        u = u[:, :rank_new].reshape(self._sBA.shape[0], d1 * rank_new)
        v = v[:rank_new, :].reshape(rank_new * d2, rank_BA)

        # factor out sAB weights from A and B
        self._A = (np.diag(1 / self._sBA) @ u).reshape(
                self._sBA.shape[0], d1, rank_new)
        self._B = (v @ np.diag(1 / self._sBA)).reshape(rank_new, d2, rank_BA)

        # new weights
        self._sAB = s_vals[:rank_new]

        if self._step % 2 == 0:
            self._sBA, self._sAB = self._sAB, self._sBA
            self._B, self._A, = self._A, self._B

        if self._rank_is_one:
            if all([self._sAB.shape[0] == 1, self._sAB.shape[-1] == 1,
                    self._sBA.shape[0] == 1, self._sBA.shape[-1] == 1]):
                # reset to initial mps if rank is still one
                self._sAB = np.ones((1))
                self._sBA = np.ones((1))
                self._A = np.ones((1, self._nu_dim, 1))
                self._B = np.ones((1, self._nu_dim, 1))
            else:
                self._rank_is_one = False
#   print(f"Effective memory depth dkmax={self._dkmax - self._step + 1}")
                if self._step == 1:
                    print("Warning: the memory cutoff dkmax may be too small"\
                            "for the given epsrel value. The algorithm may"\
                            "become unstable and inaccurate. It is recommended"\
                            "to increase dkmax until this message does no"\
                            "longer appear.")
        self._step += 1
        return self._step-1 < self.num_steps

    def update_process_tensor(self) -> None:
        """Update the process tensor. """
        assert self._step >= self._dkmax
        f = np.squeeze(np.einsum('i,ikl,l,lop->ikop', self._sAB, self._B,
                                 self._sBA, self._A))

        if self._sAB.shape[0] == 1:
            # handle trivial f
            f = np.ones((1, self._nu_dim, 1))

        # compute f[:,-1,:]^\inf = v_r * v_l^T using Lanczos
        _, v_r = eigs(f[:, -1, :], 1, which='LR')
        _, v_l = eigs(f[:, -1, :].T, 1, which='LR')
        v_l = v_l / (v_l[:, 0] @ v_r[:, 0])

        self._process_tensor.set_mpo_tensor(0, np.einsum('ia,ikj->ajk',
                                                         v_l, f[:,:-1,:]))
        self._process_tensor.set_mpo_tensor(1, np.swapaxes(f[:,:-1,:], 1, 2))
        self._process_tensor.set_cap_tensor(0, np.array([1.0]))
        self._process_tensor.set_cap_tensor(1, v_r[:, 0])
        try:
            self._process_tensor.repeating_cell = self.repeating_cell
        except Exception:
            pass

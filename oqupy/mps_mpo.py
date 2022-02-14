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
Module for MPSs and MPOs.
"""

from copy import deepcopy
from typing import Any, List, Optional, Text, Tuple

import numpy as np
from numpy import ndarray
from scipy import linalg

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype, NpDtypeReal
SystemChain = Any # from oqupy.system import SystemChain

class Gate:
    """
    Class representing an n-site gate in MPO form.

    The axes of the MPO tensors are
    (L = left bond leg, I = input leg, O = output leg, R = right bond leg):

    * for n=1: (I, O)
    * for n=2: (I, O, R), (L, I, O)
    * for n>2: (I, O, R), (L, I, O, R), ..., (L, I, O, R), (L, I, O)

    Parameters
    ----------
    sites: List[int]
        The site numbers onto which the MPO Gate acts.
    tensors: List[ndarray]
        The MPO tensors of the gate.
    """
    def __init__(
            self,
            sites: List[int],
            tensors: List[ndarray]) -> None:
        """Create a n-site gate in MPO form. """
        for site in sites:
            assert isinstance(site, int)
        for site_l, site_r in zip(sites[:-1], sites[1:]):
            assert site_l < site_r
        self._sites = sites

        assert len(sites) == len(tensors)
        self._length = len(sites)
        self._span = sites[-1] - sites[0]

        tmp_tensors = []
        for tensor in tensors:
            tmp_tensor = np.array(tensor, dtype=NpDtype)
            tmp_tensors.append(tmp_tensor)

        for i, tensor in enumerate(tmp_tensors):
            number_of_legs = 4
            number_of_legs -= 1 if i == 0 else 0
            number_of_legs -= 1 if i == self._length-1 else 0
            assert len(tensor.shape) == number_of_legs

        for tensor_l, tensor_r in zip(tmp_tensors[:-1], tmp_tensors[1:]):
            assert tensor_l.shape[-1] == tensor_r.shape[0]
        self._tensors = tmp_tensors

    def __len__(self):
        """Number of sites onto which the gate acts. """
        return self._length

    @property
    def span(self) -> int:
        """The span of sites onto which the gate acts. """
        return self._span

    @property
    def sites(self) -> List[int]:
        """The sites onto which the gate acts. """
        return self._sites

    @property
    def tensors(self) -> List[ndarray]:
        """The tensors of the MPO gate. """
        return self._tensors


class SiteGate(Gate):
    """
    An MPO gate acting on a single site.

    Parameters
    ----------
    site: int
        The site onto which the MPO gate acts.
    tensor: ndarray
        The single site MPO (which is a matrix).
    """
    def __init__(
            self,
            site: int,
            tensor: ndarray) -> None:
        """Create a single site MPO gate. """
        super().__init__([site], [tensor])


class NnGate(Gate):
    """
    An MPO gate acting on a pair of neighboring sites.

    Parameters
    ----------
    site: int
        The index of the left site.
    tensor: ndarray
        The two MPO tensors of shape.

    """
    def __init__(
            self,
            site: int,
            tensors: Tuple[(ndarray, ndarray)]) -> None:
        """Create a nearest neighbor gate. """
        super().__init__([site, site+1], [tensors[0], tensors[1]])


class GateLayer:
    """
    A layer of gates.

    Parameters
    ----------
    parallel: bool
        Whether of not the gates are suitable for a parallel application.
    gates: List[Gates]
        List of gates.
    """
    def __init__(
            self,
            parallel: bool,
            gates: List[Gate]) -> None:
        """Create a GateLayer object. """
        assert isinstance(parallel, bool)
        assert isinstance(gates, list)
        for gate in gates:
            assert isinstance(gate, Gate)

        self._parallel = parallel
        self._gates = gates

    @property
    def parallel(self) -> bool:
        """Whether of not the gates are suitable for a parallel application. """
        return self._parallel

    @property
    def gates(self) -> List[Gate]:
        """List of gates. """
        return self._gates


def compute_nn_gate(
        liouvillian: ndarray,
        site: int,
        hs_dim_l: int,
        hs_dim_r: int,
        dt: float,
        epsrel: float) -> NnGate:
    """
    Compute nearest neighbor gate from Liouvillian.

    Parameters
    ----------
    liouvillian: ndarray
        The two site Liouvillian.
    site: int
        The index of the left site.
    hs_dim_l: int
        The Hilbert space dimension of the left site.
    hs_dim_r: int
        The Hilbert space dimension of the right site.
    dt: float
        The time step length.
    epsrel: float
        The relative singular value truncation tolerance.

    Returns
    -------
    nn_gate: NnGate
        Nearest neighbor gate.
    """
    # exponentiate and transpose such that
    # axis 0 is the input and axis 1 is the output leg of the propagator.
    propagator = linalg.expm(dt*liouvillian).T
    # split leg 0 and leg 1 each into left and right.
    propagator.shape = [hs_dim_l**2,
                        hs_dim_r**2,
                        hs_dim_l**2,
                        hs_dim_r**2]
    temp = np.swapaxes(propagator, 1, 2)
    temp = temp.reshape([hs_dim_l**2 * hs_dim_l**2,
                            hs_dim_r**2 * hs_dim_r**2])
    u_full, s_full, vh_full = linalg.svd(temp, full_matrices=False)
    chi = _truncation_index(s_full, epsrel)
    s = s_full[:chi]
    u=u_full[:,:chi]
    vh=vh_full[:chi,:]
    sqrt_s = np.sqrt(s)
    u_sqrt_s = u * sqrt_s
    sqrt_s_vh =(sqrt_s * vh.T).T
    tensor_l = u_sqrt_s.reshape(hs_dim_l**2, hs_dim_l**2, chi)
    tensor_r = sqrt_s_vh.reshape(chi, hs_dim_r**2, hs_dim_r**2)

    return NnGate(site=site, tensors=(tensor_l, tensor_r))

def compute_trotter_layers(
        nn_full_liouvillians: List[ndarray],
        hs_dims: List[int],
        dt: float,
        epsrel: float) -> Tuple[GateLayer, GateLayer]:
    """
    Compute even and odd Trotter layers.

    Parameters
    ----------
    nn_full_liouvillians: List[ndarrays]
        Full list of nearest neighbor Liouvillians.
    hs_dims: List[int]
        Hilbert space dimensions of the chain sites.
    dt: float
        The time step length.
    epsrel: float
        The relative singular value truncation tolerance.

    Returns
    -------
    gate_layer_even: GateLayer
        Gate layer with nearest neighbor gates with left sites having even
        indices.
    gate_layer_odd: GateLayer
        Gate layer with nearest neighbor gates with left sites having odd
        indices.
    """
    all_gates = []
    for i, liouv in enumerate(nn_full_liouvillians):
        gate = compute_nn_gate(liouvillian = liouv,
                               site=i,
                               hs_dim_l=hs_dims[i],
                               hs_dim_r=hs_dims[i+1],
                               dt=dt,
                               epsrel=epsrel)
        all_gates.append(gate)

    gates_even = all_gates[0::2]
    gates_odd = all_gates[1::2]
    gate_layer_even = GateLayer(parallel=True, gates=gates_even)
    gate_layer_odd = GateLayer(parallel=True, gates=gates_odd)
    return [gate_layer_even, gate_layer_odd]

class TebdPropagator:
    """
    TEBD (Time Evolving Block Decimation) Propagators consist of a list of
    GateLayers.

    Parameters
    ----------
    gate_layers: List[GateLayer]
        The gate layers that make up a full time step propagation in a TEBD
        tensor network.
    """
    def __init__(
            self,
            gate_layers: List[GateLayer]) -> None:
        """Create a TebdPropagators object. """
        self._gate_layers = gate_layers

    @property
    def gate_layers(self) -> List[GateLayer]:
        """
        The gate layers that make up a full time step propagation in a TEBD
        tensor network.
        """
        return self._gate_layers


def compute_tebd_propagator(
        system_chain: SystemChain,
        time_step: float,
        epsrel: float,
        order: int) -> TebdPropagator:
    """
    Compute a TebdPropagator object for a given SystemChain.

    Parameters
    ----------
    system_chain: SystemChain
        A SystemChain object that encodes the nearest neighbor Liouvillians.
    time_step: float
        The time step length of the full TEBD propagator.
    epsrel: float
        The relative singular value truncation tolerance.
    order: int
        The expansion order.

    Returns
    -------
    tebd_propagator: TebdPropagator
        The TEBD Propagator.
    """
    nn_full_liouvillians = system_chain.get_nn_full_liouvillians()
    hs_dims = system_chain.hs_dims

    if order == 1:
        layers = compute_trotter_layers(
            nn_full_liouvillians=nn_full_liouvillians,
            hs_dims=hs_dims,
            dt=time_step,
            epsrel=epsrel)
        propagator = TebdPropagator(gate_layers=[layers[0],
                                                 layers[1]])
    elif order == 2:
        layers = compute_trotter_layers(
            nn_full_liouvillians=nn_full_liouvillians,
            hs_dims=hs_dims,
            dt=time_step/2.0,
            epsrel=epsrel)
        propagator = TebdPropagator(gate_layers=[layers[0],
                                                 layers[1],
                                                 layers[1],
                                                 layers[0]])
    else:
        raise NotImplementedError(f"Trotter layers of order {order} are " \
            + "not implemented.")

    return propagator


class AugmentedMPS(BaseAPIClass):
    """
    An augmented matrix product state (as introduced in the supplemental
    material of [Fux2022]).

    The full gamma tensors (one for each site) have the following axis:
    (L = left bond leg, P = physical leg, R = right bond leg,
    A = augmented leg).

    Depending on the rank of the input gamma tensor, it is completed with legs
    of dimension 1 according to the following interpretation of the input:

    * rank = 1: Product state vectorized density matrix, i.e. (1, P, 1, 1)
    * rank = 2: Product state density matrix, i.e. (1, p*p, 1, 1)
    * rank = 3: Canonical MPS gamma tensor, i.e. (L, P, R, 1)
    * rank = 4: Augmented MPS gamma tensor, i.e. (L, P, R, A)

    If no lambdas are given, they are assumed to be identities. A single lambda
    can be `None` (identity), a vector (giving the diagonal values) or a
    matrix (which must be diagonal).

    Parameters
    ----------
    gammas: List[ndarray]
        The input gamma tensors.
    lambdas: List[ndarray]
        The input lambda diagonal matrices.
    name: str
        An optional name for the augmented MPS.
    description: str
        An optional description of the augmented MPS.
    """
    def __init__(
            self,
            gammas: List[ndarray],
            lambdas: Optional[List[ndarray]] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create an AugmentedMPS object. """
        # input parsing
        self._n = len(gammas)
        assert self._n > 0
        if lambdas is not None:
            assert len(lambdas) == len(gammas)-1

        tmp_gammas = []
        for g in gammas:
            tmp_gamma = np.array(g, dtype=NpDtype)
            shape = deepcopy(tmp_gamma.shape)
            rank = len(shape)
            if rank == 4:
                pass
            elif rank == 3:
                tmp_gamma.shape = (shape[0], shape[1], shape[2], 1)
            elif rank == 2:
                tmp_gamma.shape = (1, shape[0]*shape[1], 1, 1)
            elif rank == 1:
                tmp_gamma.shape = (1, shape[0], 1, 1)
            else:
                raise ValueError()
            tmp_gammas.append(tmp_gamma)

        bond_dims = []
        for g1, g2 in zip(tmp_gammas[:-1], tmp_gammas[1:]):
            assert g1.shape[3] == g2.shape[0]
            bond_dims.append(g1.shape[3])

        if lambdas is None:
            lambdas = [None] * (self._n - 1)

        tmp_lambdas = []
        for bond_dim, l in zip(bond_dims, lambdas):
            if l is None:
                tmp_lambda = np.ones(bond_dim, dtype=NpDtypeReal)
            else:
                tmp_lambda = np.array(l, dtype=NpDtypeReal)
                shape = tmp_lambda.shape
                rank = len(shape)
                if rank == 2:
                    assert np.all(
                        tmp_lambda == np.diag(np.diagonal(tmp_lambda)))
                    tmp_lambda = np.diagonal(tmp_lambda)
                elif rank == 1:
                    pass
                else:
                    raise ValueError()
                assert np.all(tmp_lambda > 0.0), \
                    "All lambda matrix diagonal values must be positive. "
                assert len(tmp_lambda) == bond_dim
            tmp_lambdas.append(tmp_lambda)

        self._gammas = tmp_gammas
        self._lambdas = tmp_lambdas
        super().__init__(name, description)

    @property
    def gammas(self) -> ndarray:
        """"The gamma tensors."""
        return self._gammas

    @property
    def lambdas(self) -> ndarray:
        """"The values of the lambda matrices diagonals."""
        return self._lambdas


def _truncation_index(s: ndarray, epsrel: float) -> int:
    """Helper function to figure out the right singular value cutoff. """
    absrel = s[0] * epsrel
    cummulative_square = np.cumsum(np.flip(s)**2)
    chi = np.count_nonzero(cummulative_square > absrel**2)
    return chi

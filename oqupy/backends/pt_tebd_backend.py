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
Module for Process Tensor Time Evolving Block Decimation (PT-TEBD) backend.
The algorithms in this module are explained in the supplemental material
of [Fux2022].

**[Fux2022]**
G. E. Fux, D. Kilda, B. W. Lovett, and J. Keeling, *Thermalization of a
spin chain strongly coupled to its environment*, arXiv:2201.05529 (2022).

"""

import concurrent
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray
import tensornetwork as tn

from oqupy.config import NpDtype
from oqupy.process_tensor import BaseProcessTensor
from oqupy.mps_mpo import GateLayer, SiteGate, NnGate

NoneType = type(None)

r"""
The augmented MPS has the form

          2 1                   2 1               2 1
           \|                    \|                \|
 0--λ--1 0--Γ--3  0--λ--1 ...  0--Γ--3  0--λ--1  0--Γ--3 0--λ--1

where the first and last λ = [[1.0]] are attached for convenience.
"""

class PtTebdBackend:
    """
    The Process Tensor Time Evolving Block Decimation backend employing
    the tensornetwork package.

    Parameters
    ----------
    gammas: List[ndarray]
        Full initial gamma tensors.
    lambdas: List[ndarray]
        The diagonals of the initial lambda matrices.
    esprel: float
        The relative singular value truncation tolerance.
    config: Dict
        Backend configuration dictionary.

    Backend configuration `config` allows for the following options:
    * 'parallel' : 'multiprocess' / 'multithread'
    """
    def __init__(
            self,
            gammas: List[ndarray],
            lambdas: List[ndarray],
            epsrel: float,
            config: Dict) -> None:
        """Create a PtTebdBackend. """
        assert len(gammas) == len(lambdas) + 1
        self._n = len(gammas)
        self._epsrel = epsrel
        self._config = config

        if "parallel" in config:
            self._parallel = config["parallel"]
        else:
            self._parallel = None

        self._gammas = []
        self._lambdas = []
        self._left_e = None
        self._right_e = None
        self._gam_lam_es = []
        self._lam_gam_es = []
        self._phys_es = []
        self._pt_es = []

        for gam in gammas:
            self._gammas.append(tn.Node(gam))
        for gam in self._gammas:
            self._phys_es.append(gam[1])
            self._pt_es.append(gam[2])

        left_dim = gammas[0].shape[0]
        right_dim = gammas[-1].shape[3]
        self._lambdas.append(tn.Node(np.identity(left_dim, dtype=NpDtype)))
        for lam in lambdas:
            self._lambdas.append(tn.Node(np.diag(lam)))
        self._lambdas.append(tn.Node(np.identity(right_dim, dtype=NpDtype)))

        self._left_e = self._lambdas[0][0]
        self._right_e = self._lambdas[-1][1]
        for i in range(self._n):
            self._lam_gam_es.append(self._lambdas[i][1] ^ self._gammas[i][0])
            self._gam_lam_es.append(self._gammas[i][3] ^ self._lambdas[i+1][0])

        self._bath_trace_gammas = None
        self._full_trace_gammas = None
        self._total_left_traces = None
        self._total_right_traces = None
        self._total_trace = None

    @property
    def n(self) -> int:
        """Number of chain sites. """
        return self._n

    def get_gamma(self, site: int) -> ndarray:
        """Return gamma tensor of site. """
        return self._gammas[site].get_tensor()

    def get_lambda(self, site: int) -> ndarray:
        """Return lambda matrix of site. """
        return self._lambdas[site+1].get_tensor()

    def get_bond_dimensions(self) -> ndarray:
        """Return bond dimensions of the chain. """
        bond_dimensions = []
        for gam_lam_e in self._gam_lam_es[:-1]:
            bond_dimensions.append(gam_lam_e.dimension)
        return np.array(bond_dimensions)

    def apply_nn_gate_layer(self, gate_layer: GateLayer):
        """Apply a nearest neighbor gate layer to the augmented MPS. """
        if self._parallel is None:
            for gate in gate_layer.gates:
                self.apply_nn_gate(gate)
        else:
            input_datas = []
            for gate in gate_layer.gates:
                input_datas.append(self._apply_nn_gate_get_data(gate))
            if self._parallel == "multiprocess":
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    output_datas = executor.map(apply_nn_gate, input_datas)
            elif self._parallel == "multithread":
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    output_datas = executor.map(apply_nn_gate, input_datas)
            else:
                raise NotImplementedError("Parallelisation method " \
                    + f"'{self._parallel}' is not implementedds!")
            for output_data in output_datas:
                self._apply_nn_gate_replace_gam_lam_gam(*output_data)


    def apply_process_tensors(
            self,
            step: int,
            process_tensors: List[BaseProcessTensor]):
        """Apply the next bits of the process tensors to the augmented MPS. """
        for site in range(self.n):
            pt_tensor = process_tensors[site].get_mpo_tensor(step-1)
            if pt_tensor is not None:
                pt = tn.Node(pt_tensor)
                pt[0] ^ self._pt_es[site]
                pt[2] ^ self._phys_es[site]
                self._pt_es[site] = pt[1]
                self._phys_es[site] = pt[3]
                gam = self._gammas[site] @ pt
                gam.reorder_edges([self._lam_gam_es[site],
                                self._phys_es[site],
                                self._pt_es[site],
                                self._gam_lam_es[site]])
                self._gammas[site] = gam

    def _tidy_edge_order(self):
        """Bring gamma edges into order. """
        for site in range(self.n):
            self._gammas[site].reorder_edges(
                [self._lam_gam_es[site],
                self._phys_es[site],
                self._pt_es[site],
                self._gam_lam_es[site]])

    def _apply_nn_gate_get_data(self,  gate: NnGate):
        """Get all data to apply a nearest neighbor gate. """
        site_l = gate.sites[0]
        site_r = gate.sites[1]
        gate_l = tn.Node(gate.tensors[0])
        gate_r = tn.Node(gate.tensors[1])
        lam_l = self._lambdas[site_l].copy()
        gam_l = self._gammas[site_l].copy()
        lam_m = self._lambdas[site_l+1].copy()
        gam_r = self._gammas[site_r].copy()
        lam_r = self._lambdas[site_r+1].copy()
        data = (site_l, lam_l, gam_l, lam_m, gam_r,
                lam_r, gate_l, gate_r, self._epsrel)
        return data

    def _apply_nn_gate_replace_gam_lam_gam(
            self,
            site_l,
            new_gam_l,
            new_lam_m,
            new_gam_r):
        """
        Replace gamma-lambda-gamma part of the augmented MPS with new
        tensors after the application of the nearest neighbor gate.
        """
        site_r = site_l + 1
        tn.remove_node(self._gammas[site_l])
        tn.remove_node(self._gammas[site_r])
        tn.remove_node(self._lambdas[site_l+1])
        self._gammas[site_l] = new_gam_l
        self._gammas[site_r] = new_gam_r
        self._lambdas[site_l+1] = new_lam_m
        self._phys_es[site_l] = new_gam_l[1]
        self._phys_es[site_r] = new_gam_r[1]
        self._pt_es[site_l] = new_gam_l[2]
        self._pt_es[site_r] = new_gam_r[2]
        self._lam_gam_es[site_l] = self._lambdas[site_l][1] \
                                   ^ new_gam_l[0]
        self._gam_lam_es[site_l] = new_gam_l[3] ^ new_lam_m[0]
        self._lam_gam_es[site_r] = new_lam_m[1] ^ new_gam_r[0]
        self._gam_lam_es[site_r] = new_gam_r[3] \
                                   ^ self._lambdas[site_r+1][0]

    def apply_nn_gate(self, gate: NnGate):
        """Apply a nearest neighbor gate on the augmented MPS. """
        assert isinstance(gate, NnGate)
        data = self._apply_nn_gate_get_data(gate)
        new_gam_lam_gam = _apply_nn_gate(*data)
        self._apply_nn_gate_replace_gam_lam_gam(*new_gam_lam_gam)

    def apply_site_gate_layer(self, gate_layer: GateLayer) -> None:
        """Apply a single site gate layer on the augmented MPS. """
        for gate in gate_layer.gates:
            self.apply_site_gate(gate)

    def apply_site_gate(self, gate: SiteGate) -> None:
        """Apply a single site gate on the augmented MPS. """
        assert isinstance(gate, SiteGate)
        site = gate.sites[0]
        matrix = tn.Node(gate.tensors[0])
        matrix[1] ^ self._phys_es[site]
        self._phys_es[site] = matrix[0]
        gam = self._gammas[site] @ matrix
        gam.reorder_edges([self._lam_gam_es[site],
                           self._phys_es[site],
                           self._pt_es[site],
                           self._gam_lam_es[site]])
        self._gammas[site] = gam

    def clear_traces(self):
        """Remove temporarily computed traces of the augmented MPS. """
        self._bath_trace_gammas = None
        self._full_trace_gammas = None
        self._total_left_traces = None
        self._total_right_traces = None
        self._total_trace = None

    def compute_traces(
            self,
            step: int,
            process_tensors: List[BaseProcessTensor]):
        """Compute current traces of the augmented MPS. """
        self.clear_traces()
        self._compute_bath_trace_gammas(step, process_tensors)
        self._compute_full_trace_gammas()
        self._compute_total_traces()

    def _compute_bath_trace_gammas(
            self,
            step: int,
            process_tensors: List[BaseProcessTensor]):
        """
        Trace out the bath correlations of the augmented MPS to yield a
        canonical MPS. See Fig. S2(k-l) in the supplemental material of
        [Fux2022].
        """
        node_dict, edge_dict = tn.copy(self._gammas)
        self._bath_trace_gammas = []
        for site in range(self._n):
            pt_cap = tn.Node(process_tensors[site].get_cap_tensor(step))
            edge_dict[self._pt_es[site]] ^ pt_cap[0]
            bath_trace_gamma = node_dict[self._gammas[site]] @ pt_cap
            bath_trace_gamma.reorder_edges(
                [edge_dict[self._lam_gam_es[site]],
                 edge_dict[self._phys_es[site]],
                 edge_dict[self._gam_lam_es[site]]])
            self._bath_trace_gammas.append(bath_trace_gamma)

    def _compute_full_trace_gammas(self):
        """
        Compute all local traces over the gamma matrices as shown at the top of
        Fig. S1(d) in the supplemental material of [Fux2022].
        """
        assert self._bath_trace_gammas is not None

        node_dict, edge_dict = tn.copy(self._bath_trace_gammas)
        self._full_trace_gammas = []

        for site in range(self._n):
            lam_gam_edge = edge_dict[self._bath_trace_gammas[site][0]]
            phys_edge = edge_dict[self._bath_trace_gammas[site][1]]
            gam_lam_edge = edge_dict[self._bath_trace_gammas[site][2]]

            phys_edge_dim = self._bath_trace_gammas[site].get_dimension(1)
            hs_dim = _isqrt(phys_edge_dim)
            trace_cap = tn.Node(
                np.identity(hs_dim, dtype=NpDtype).reshape(hs_dim**2))

            phys_edge ^ trace_cap[0]
            full_trace_gamma = node_dict[self._bath_trace_gammas[site]] \
                               @ trace_cap
            full_trace_gamma.reorder_edges([lam_gam_edge, gam_lam_edge])
            self._full_trace_gammas.append(full_trace_gamma)

    def _compute_total_traces(self):
        """Compute the total trace of the left and right parts of the chain. """
        assert self._full_trace_gammas is not None

        nodes = self._full_trace_gammas + self._lambdas
        node_dict, edge_dict = tn.copy(nodes)

        temp_left = tn.Node(np.array([1.0], dtype=NpDtype))
        temp_left[0] ^ edge_dict[self._left_e]
        total_left_trace =  temp_left @ node_dict[self._lambdas[0]]
        self._total_left_traces = [total_left_trace.copy()]
        for site in range(self._n):
            tr_gam = node_dict[self._full_trace_gammas[site]]
            lam = node_dict[self._lambdas[site+1]]
            total_left_trace[0] ^ tr_gam[0]
            tr_gam[1] ^ lam[0]
            total_left_trace = total_left_trace @ tr_gam @ lam
            if site == self._n - 1:
                temp_right = tn.Node(np.array([1.0], dtype=NpDtype))
                total_left_trace[0] ^ temp_right[0]
                self._total_trace = (total_left_trace @ temp_right).get_tensor()
            else:
                self._total_left_traces.append(total_left_trace.copy())

        nodes = self._full_trace_gammas + self._lambdas
        node_dict, edge_dict = tn.copy(nodes)

        temp_right = tn.Node(np.array([1.0], dtype=NpDtype))
        edge_dict[self._right_e] ^ temp_right[0]
        total_right_trace =  node_dict[self._lambdas[-1]] @ temp_right
        self._total_right_traces = [total_right_trace.copy()]
        for site in reversed(range(1, self._n)):
            tr_gam = node_dict[self._full_trace_gammas[site]]
            lam = node_dict[self._lambdas[site]]
            tr_gam[1] ^ total_right_trace[0]
            lam[1] ^ tr_gam[0]
            total_right_trace = total_right_trace @ tr_gam @ lam
            self._total_right_traces.insert(0,total_right_trace.copy())

    def get_norm(self) -> complex:
        """Get the total trace of the current chain state. """
        return complex(self._total_trace)

    def get_site_density_matrix(self, site: int) -> ndarray:
        """Extract the reduced density matrix of a single site. """
        gam = self._bath_trace_gammas[site].copy()
        left_tr = self._total_left_traces[site].copy()
        right_tr = self._total_right_traces[site].copy()

        left_tr[0] ^ gam[0]
        gam[2] ^ right_tr[0]
        dm_node = left_tr @ gam @ right_tr
        dm_vector = dm_node.get_tensor()
        hs_dim = _isqrt(dm_vector.shape[0])
        density_matrix = dm_vector.reshape(hs_dim, hs_dim)
        return density_matrix

    def get_density_matrix(self, sites: List[int]) -> ndarray:
        """Extract the reduced density matrix of a list of sites. """
        assert isinstance(sites, list)
        assert len(sites) >= 1
        if len(sites) == 1:
            return self.get_site_density_matrix(sites[0])

        assert sites == sorted(sites)
        s = sorted(sites)

        left_tr = self._total_left_traces[s[0]].copy()
        right_tr = self._total_right_traces[s[-1]].copy()
        gammas = []
        for a, b  in zip(s[:-1],s[1:]):
            gam = self._bath_trace_gammas[a].copy()
            m = self._lambdas[a+1].copy()
            for i in range(a+1, b):
                gam_tr = self._full_trace_gammas[i].copy()
                lam = self._lambdas[i+1].copy()
                m[1] ^ gam_tr[0]
                gam_tr[1] ^ lam[0]
                m = m @ gam_tr @ lam
            gam[2] ^ m[0]
            gam = gam @ m
            gammas.append(gam)
        gam = self._bath_trace_gammas[s[-1]].copy()
        gammas.append(gam)

        dm_edges = []
        left_tr[0] ^ gammas[0][0]
        dm_edges.append(gammas[0][1])
        dm_bond_edge = gammas[0][2]
        dm = left_tr @ gammas[0]

        for gam in gammas[1:]:
            dm_bond_edge ^ gam[0]
            dm_edges.append(gam[1])
            dm_bond_edge = gam[2]
            dm = dm @ gam

        dm_bond_edge ^ right_tr[0]
        dm = dm @ right_tr

        dm_left_edges = []
        dm_right_edges = []

        for dm_edge in dm_edges:
            dim = _isqrt(dm_edge.dimension)
            split_edges = tn.split_edge(dm_edge, (dim,dim))
            assert len(split_edges) == 2
            dm_left_edges.append(split_edges[0])
            dm_right_edges.append(split_edges[1])

        left_edge = tn.flatten_edges(dm_left_edges)
        right_edge = tn.flatten_edges(dm_right_edges)
        dm.reorder_edges([left_edge, right_edge])

        return dm.get_tensor()


def apply_nn_gate(input_data):
    """
    Apply a nearest neighbor gate to the augmented MPS.

    This algorithm is shown in the Figs. S2(c-h) in the supplemental material
    of [Fux2022].
    """
    return _apply_nn_gate(*input_data)

def _apply_nn_gate(
        site_l: int,
        lam_l: tn.Node,
        gam_l: tn.Node,
        lam_m: tn.Node,
        gam_r: tn.Node,
        lam_r: tn.Node,
        gate_l: tn.Node,
        gate_r: tn.Node,
        epsrel: float) -> Tuple[int, tn.Node, tn.Node, tn.Node]:
    """
    Apply a nearest neighbor gate to the augmented MPS.

    This algorithm is shown in the Figs. S2(c-h) in the supplemental material
    of [Fux2022].
    """

    # -- compute inverted left and right lambdas --
    inv_lam_l = _invert_lambda(lam_l)
    inv_lam_r = _invert_lambda(lam_r)

    # -- contraction left nodes and right nodes --
    left_e = lam_l[0]
    lam_l[1] ^ gam_l[0]
    phy_l_gate_l_e = gam_l[1] ^ gate_l[1]
    new_phy_l_e = gate_l[0]
    pt_l_e = gam_l[2]
    gam_l_lam_m_e = gam_l[3] ^ lam_m[0]
    gate_l[2] ^ gate_r[0]
    lam_m_gam_r_e = lam_m[1] ^ gam_r[0]
    phy_r_gate_r_e = gam_r[1] ^ gate_r[2]
    new_phy_r_e = gate_r[1]
    pt_r_e = gam_r[2]
    gam_r[3] ^ lam_r[0]
    right_e = lam_r[1]

    left_temp_node = lam_l @ gam_l
    right_temp_node =  gam_r @ lam_r

    # -- split off the process tensor legs --
    u, s, vh, _ = tn.split_node_full_svd(
        node=left_temp_node,
        left_edges=[left_e, pt_l_e],
        right_edges=[phy_l_gate_l_e, gam_l_lam_m_e],
        max_truncation_err=epsrel,
        relative=True)
    left_bond_e = s[0]
    left_temp_node = u
    left_middle_temp_node = s @ vh

    u, s, vh, _ = tn.split_node_full_svd(
        node=right_temp_node,
        left_edges=[lam_m_gam_r_e, phy_r_gate_r_e],
        right_edges=[pt_r_e, right_e],
        max_truncation_err=epsrel,
        relative=True)
    right_bond_e = s[1]
    right_temp_node = vh
    right_middle_temp_node = u @ s

    # -- apply gate and contract to one tensor --
    theta_node = tn.contractors.optimal(
        nodes=[
            left_middle_temp_node,
            gate_l,
            lam_m,
            gate_r,
            right_middle_temp_node],
        output_edge_order=[
            left_bond_e,
            new_phy_l_e,
            new_phy_r_e,
            right_bond_e])

    u, s, vh, _ = tn.split_node_full_svd(
        node=theta_node,
        left_edges=[left_bond_e, new_phy_l_e],
        right_edges=[new_phy_r_e, right_bond_e],
        max_truncation_err=epsrel,
        relative=True)
    new_temp_lm = u
    new_lam_m = s
    new_temp_mr = vh
    new_gam_l_lam_m_e = new_lam_m[0]
    new_lam_m_gam_r_e = new_lam_m[1]

    # -- contract with inverted left and right lambdas --
    new_left_e = inv_lam_l[0]
    inv_lam_l[1] ^ left_e
    new_right_e = inv_lam_r[1]
    right_e ^ inv_lam_r[0]

    new_gam_l = tn.contractors.optimal(
        nodes=[
            inv_lam_l,
            left_temp_node,
            new_temp_lm],
        output_edge_order=[
            new_left_e,
            new_phy_l_e,
            pt_l_e,
            new_gam_l_lam_m_e])

    new_gam_r = tn.contractors.optimal(
        nodes=[
            new_temp_mr,
            right_temp_node,
            inv_lam_r],
        output_edge_order=[
            new_lam_m_gam_r_e,
            new_phy_r_e,
            pt_r_e,
            new_right_e])

    for node in [new_gam_l, new_lam_m, new_gam_r]:
        for edge in node.get_all_nondangling():
            edge.disconnect()

    return site_l, new_gam_l, new_lam_m, new_gam_r


def _invert_lambda(node: tn.Node) -> tn.Node:
    """Invert a diagonal lambda matrix. """
    tensor = node.get_tensor()
    assert _is_diagonal_matrix(tensor)
    diagonal = tensor.diagonal()
    return tn.Node(np.diag(1/diagonal))

def _is_diagonal_matrix(tensor: ndarray):
    """Check if a matrix is diagonal. """
    assert len(tensor.shape) == 2
    i, j = tensor.shape
    assert i == j
    test = tensor.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def _isqrt(x):
    """Check if an integer is a perfect square. """
    n = int(round(np.sqrt(x)))
    if n**2 != x:
        raise ValueError(
            "{x} is not a perfect square. Can't take integer square root!")
    return n

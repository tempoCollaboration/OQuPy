# Implementation of iTEBD-TEMPO
# Author: Valentin Link (valentin.link@tu-dresden.de)

import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import expm, norm, svd
from scipy.sparse.linalg import eigs
from typing import Callable, Optional
from tqdm import tqdm
from ncon import ncon  # ncon performs better than np.einsum


def iTEBD_apply_gate(gate: np.ndarray, A: np.ndarray, sAB: np.ndarray, B: np.ndarray, sBA: np.ndarray, rank: int, rtol: float, ctol: Optional[float] = 1e-13):
    """
    single iTEBD step, scheme adapted from https://www.tensors.net/mps
    :param gate: TEBD gate for A-B link
    :param A: A tensor (left)
    :param sAB: weight for A-B link
    :param B: B tensor (right)
    :param sBA: weight for B-A link
    :param rank: maximum rank in svd compression
    :param rtol: relative error for svd compression
    :param ctol: cutoff for weights sBA which need to be inverted
    :return: new tensors and weights A, sAB, B, sBA
    """

    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    # ensure weights are above tolerance (needed for inversion)
    sBA[np.abs(sBA) < ctol] = ctol

    # MPS - gate contraction
    d1 = gate.shape[1]
    d2 = gate.shape[-1]
    rank_BA = sBA.shape[0]

    u, s_vals, v = svd(ncon([np.diag(sBA), A, np.diag(sAB), B, np.diag(sBA), gate], [[-1, 1], [1, 5, 2], [2, 4], [4, 6, 3], [3, -4], [5, -2, 6, -3]]).reshape([d1 * rank_BA, d2 * rank_BA]), full_matrices=False)

    # truncate singular values
    if rtol is None:
        rank_new = min(rank, len(s_vals))
    else:
        s_vals_sum = np.cumsum(s_vals) / np.sum(s_vals)
        rank_rtol = np.searchsorted(s_vals_sum, 1 - rtol) + 1
        rank_new = min(rank, len(s_vals), rank_rtol)
    u = u[:, :rank_new].reshape(sBA.shape[0], d1 * rank_new)
    v = v[:rank_new, :].reshape(rank_new * d2, rank_BA)

    # factor out sAB weights from A and B
    A = (np.diag(1 / sBA) @ u).reshape(sBA.shape[0], d1, rank_new)
    B = (v @ np.diag(1 / sBA)).reshape(rank_new, d2, rank_BA)

    # new weights
    sAB = s_vals[:rank_new]

    return A, sAB, B, sBA


class BathCorrelation():
    """ A class to store the bath correlation function. """

    def __init__(self, bcf: Callable[[float], complex]):
        """
        :param bcf: The bath correlation function.
        """

        self.bcf = bcf

    def compute_eta(self, n: int, delta: float) -> np.ndarray:
        """
        Computes the discretized bath correlation function (eta) for n time steps delta.
        :param n: Number of time steps.
        :param delta: Time step.
        :return: Discretized bath correlation function.
        """

        eta = np.zeros(n, dtype=np.complex128)
        eta[0] += dblquad(lambda s, t: np.real(self.bcf(t - s)), 0, delta, lambda t: 0, lambda t: t)[0]
        eta[0] += dblquad(lambda s, t: np.imag(self.bcf(t - s)), 0, delta, lambda t: 0, lambda t: t)[0] * 1j

        for k in range(1, n):
            eta[k] += dblquad(lambda s, t: np.real(self.bcf(t - s)), k * delta, (k + 1) * delta, 0, delta)[0]
            eta[k] += dblquad(lambda s, t: np.imag(self.bcf(t - s)), k * delta, (k + 1) * delta, 0, delta)[0] * 1j

        return eta


class iTEBD_TEMPO():
    """ A class to compute and approximate the influence functional unsing iTEBD-TEMPO and compute dynamics. """

    def __init__(self, s_vals: np.ndarray, delta: float, bcf: Callable[[float], complex], n_c: int):
        """
        :param s_vals: Real eigenvalues of the system-bath coupling operator.
        :param delta: time step for Trotter splitting.
        :param bcf: Bath correlation function.
        :param n_c: Memory cutoff. Should be chosen large enough.
        """

        self.n_c = n_c
        self.n_c_eff = n_c
        self.s_vals = s_vals
        self.s_dim = self.s_vals.size
        self.nu_dim = self.s_vals.size ** 2 + 1
        self.bcf = BathCorrelation(bcf)
        self.delta = delta
        self.eta = self.bcf.compute_eta(self.n_c, delta)
        self.s_diff = np.empty((self.nu_dim - 1), dtype=np.complex128)
        self.s_sum = np.empty((self.nu_dim - 1), dtype=np.complex128)
        for nu in range(self.nu_dim - 1):
            i, j = int(nu / self.s_dim), nu % self.s_dim
            self.s_diff[nu] = self.s_vals[i] - self.s_vals[j]
            self.s_sum[nu] = self.s_vals[i] + self.s_vals[j]
        self.s_diff = np.pad(self.s_diff, [(0, 1)])
        self.s_sum = np.pad(self.s_sum, [(0, 1)])
        self.kron_delta = np.identity(self.nu_dim)
        self.f = None
        return

    def compute_f(self, rtol: float, rank: Optional[int] = np.inf):
        """
        Compute the infinite influence functional tensor f using iTEBD.

        :param rtol: Relative tolerance for svd compression.
        :param rank: Maximum allowed rank (bond dimension).
        """

        A = np.ones((1, self.nu_dim, 1))
        B = np.ones((1, self.nu_dim, 1))
        sAB = np.ones((1))
        sBA = np.ones((1))
        rank_is_one = True

        for k in tqdm(range(1, self.n_c + 1), desc='building influence functional'):
            i_tens = np.exp(-self.eta[self.n_c - k].real * np.outer(self.s_diff, self.s_diff) - 1j * self.eta[self.n_c - k].imag * np.outer(self.s_sum, self.s_diff))

            if k == self.n_c:
                gate = np.einsum('a,ij,jb,j->jabi', np.ones((1)), self.kron_delta, self.kron_delta, np.diagonal(i_tens))
            else:
                gate = np.einsum('ij,ab,aj->jabi', self.kron_delta, self.kron_delta, i_tens)

            if k % 2 == 0:
                B, sBA, A, sAB = iTEBD_apply_gate(gate, B, sBA, A, sAB, rank, rtol=rtol)
            else:
                A, sAB, B, sBA = iTEBD_apply_gate(gate, A, sAB, B, sBA, rank, rtol=rtol)

            if rank_is_one:
                if np.alltrue([sAB.shape[0] == 1, sAB.shape[-1] == 1, sBA.shape[0] == 1, sBA.shape[-1] == 1]):
                    # reset to initial mps if rank is still one
                    sAB = np.ones((1))
                    sBA = np.ones((1))
                    A = np.ones((1, self.nu_dim, 1))
                    B = np.ones((1, self.nu_dim, 1))
                else:
                    rank_is_one = False
                    self.n_c_eff = self.n_c - k + 1
                    if k == 1:
                        print('Warning: the memory cutoff n_c may be too small for the given rtol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear.')
        self.f = np.squeeze(ncon([np.diag(sAB), B, np.diag(sBA), A], [[-1, 1], [1, -2, 2], [2, 3], [3, -3, -4]]))

        if sAB.shape[0] == 1:
            # handle trivial f
            self.f = np.ones((1, self.nu_dim, 1))

        # compute f[:,-1,:]^\inf = v_r * v_l^T using Lanczos
        w, v_r = eigs(self.f[:, -1, :], 1, which='LR')
        w, v_l = eigs(self.f[:, -1, :].T, 1, which='LR')
        self.v_r = v_r[:, 0]
        self.v_l = v_l[:, 0] / (v_l[:, 0] @ v_r[:, 0])

        print('rank ', self.f.shape[0])
        return

    def get(self, i_path: np.ndarray) -> complex:
        """ Computes the influence of a single path with indices i_path.

        :param i_path: Array of indices at which the generated influence functional should be evaluated.
        :return: Value of the influence functional along the path.
        """
        assert self.f is not None, "the influence functional has not yet been computed, run self.compute_f first"
        assert type(i_path[0]) is np.int32, "i_path must be an integer numpy array"
        val = 1. * self.v_l
        for i in range(i_path.size):
            val = val @ self.f[:, i_path[i], :]
        return val @ self.v_r

    def get_exact(self, i_path: np.ndarray) -> complex:
        """ Computes the exact influence of a single path with indices i_path. 
        
        :param i_path: Array of indices at which the influence functional should be evaluated.
        :return: Exact value of the influence functional along the path.
        """
        assert type(i_path[0]) is np.int32, "i_path must be an integer numpy array"
        n = i_path.size
        if n > self.eta.size:
            self.eta = self.bcf.compute_eta(n, self.delta)
        s_diff_path = [self.s_diff[i] for i in np.flip(i_path)]
        s_sum_path = [self.s_sum[i] for i in np.flip(i_path)]
        are = np.concatenate((np.flip(self.eta[:n].real), np.zeros(n)))
        aim = np.concatenate((np.flip(self.eta[:n].imag), np.zeros(n)))
        return np.exp(-np.dot(s_diff_path, np.convolve(are, s_diff_path, 'valid')[:n]) - 1j * np.dot(s_diff_path, np.convolve(aim, s_sum_path, 'valid')[:n]))

    def evolve(self, h_s: np.ndarray, rho_0: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the time evolution for n time steps.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :param rho_0: System initial state in the eigenbasis of the coupling operator.
        :param n: Number of time-steps for the propagation.
        :return: Time evolution of density matrix.
        """

        assert self.f is not None, "the influence functional has not yet been computed, run self.compute_f first"

        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)

        rho_t = np.empty((n + 1, self.s_dim, self.s_dim), dtype=np.complex128)
        rho_t[0] = rho_0

        evol_tens = ncon([self.f[:, :-1, :], u], [[-1, 2, -3], [-2, 2, -4]])
        state = ncon([self.v_l, rho_0.flatten()], [[-2], [-3]])

        for i in tqdm(range(n), desc='time evolution running'):
            state = ncon([state, evol_tens], [[1, 2], [1, 2, -2, -3]])
            rho_t[i + 1] = ncon([self.v_r, state], [[1], [1, -1]]).reshape(rho_0.shape)
        return rho_t

    def steadystate(self, h_s: np.ndarray) -> np.ndarray:
        """
        Compute the steady state using Lanczos.

        :param h_s: System Hamiltonian in the eigenbasis of the coupling operator.
        :return: Steady state density matrix.
        """

        assert self.f is not None, "the influence functional has not yet been computed, run self.compute_f first"

        liu_s = np.kron(expm(-1j * h_s * self.delta / 2), expm(1j * h_s * self.delta / 2).T)
        u = np.einsum('ab,bc->abc', liu_s.T, liu_s.T)

        evol_tens = ncon([self.f[:, :-1, :], u], [[-1, 1, -3], [-2, 1, -4]])

        w, v = eigs(evol_tens.reshape([evol_tens.shape[0] * evol_tens.shape[1], evol_tens.shape[2] * evol_tens.shape[3]]).T, 1, which='LR')

        rho_ss = (self.v_r @ v.reshape([self.v_r.size, self.s_dim**2])).reshape([self.s_dim, self.s_dim])

        return rho_ss / np.trace(rho_ss)

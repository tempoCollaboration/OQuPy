# Copyright 2026 The OQuPy Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finite-mode Automated Compression of Environments (ACE) backend.

The influence functional has one physical index
q = ket * system_dimension + bra per time interval. Each tensor has the
native OQuPy shape (bond_in, bond_out, q). The environment initial states
and final traces are absorbed in the first and last tensors.

Independent modes sharing one diagonal system coupling contribute an
elementwise product of influence functionals. Merging them therefore takes
Kronecker products of the bonds, with a shared physical index. SVD sweeps
compress these products. This ports the mode-product/preselection approach
of the supplied ACE prototype, replacing its four-index layout with OQuPy's
diagonal three-index representation.

The relative singular-value cutoff is local, not a bound on the error of
a final observable. Preselection additionally discards pairs using products
of the two inputs' relative bond spectra; those products are a heuristic
for a shared-physical-index product, not its exact Schmidt spectrum.
"""

import numpy as np
from scipy import linalg
from oqupy.util import get_progress


def _svd(matrix):
    """Use a reduced SVD, with the robust LAPACK driver as a fallback."""
    try:
        return linalg.svd(matrix, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        return linalg.svd(
            matrix, full_matrices=False, check_finite=False,
            lapack_driver="gesvd")


def _retained_rank(singular_values, epsrel):
    """Keep s > epsrel*s[0], at least one; zero tolerance keeps all s."""
    if epsrel == 0.0:
        return len(singular_values)
    return max(
        1, int(np.count_nonzero(
            singular_values > epsrel * singular_values[0])))


def _right_canonicalize(tensors, advance=None):
    """Right-canonicalize, allowing a scalar factor on each isometry.

    Distributing scalar factors across sites avoids accumulating the full
    influence functional's exponentially large Frobenius norm at one site.
    These factors do not change relative singular values at any cut.
    """
    for site in range(len(tensors) - 1, 0, -1):
        tensor = tensors[site]
        left, right, physical = tensor.shape
        matrix = tensor.reshape(left, right * physical)

        q_matrix, r_matrix = linalg.qr(
            matrix.conj().T, mode="economic", check_finite=False)
        transfer = r_matrix.conj().T
        scale = float(np.max(np.abs(transfer)))
        if scale == 0.0:
            scale = 1.0
        tensors[site] = (
            q_matrix.conj().T.reshape(-1, right, physical) * scale)
        tensors[site - 1] = np.einsum(
            "abq,bc->acq", tensors[site - 1], transfer / scale,
            optimize=True)
        if advance is not None:
            advance()


def _left_sweep(tensors, epsrel, advance=None):
    """SVD-compress left to right and return relative spectra at each cut.

    A right-canonical input makes each local singular spectrum proportional
    to that of the full influence functional at the same cut. Leading
    singular values are left on individual sites for numerical stability.
    """
    spectra = []
    for site in range(len(tensors) - 1):
        tensor = tensors[site]
        left, right, physical = tensor.shape
        matrix = tensor.transpose(0, 2, 1).reshape(left * physical, right)
        u_matrix, singular_values, vh_matrix = _svd(matrix)
        rank = _retained_rank(singular_values, epsrel)
        singular_values = singular_values[:rank]
        scale = float(singular_values[0])
        if scale == 0.0:
            scale = 1.0
        relative_values = singular_values / scale
        tensors[site] = (
            u_matrix[:, :rank].reshape(left, physical, rank)
            .transpose(0, 2, 1) * scale)
        transfer = relative_values[:, None] * vh_matrix[:rank, :]
        tensors[site + 1] = np.einsum(
            "ab,bcq->acq", transfer, tensors[site + 1], optimize=True)
        spectra.append(relative_values.copy())
        if advance is not None:
            advance()
    return spectra


def _compress(tensors, epsrel, advance=None):
    """Compress an owned tensor list in place without dense path expansion."""
    _right_canonicalize(tensors, advance)
    _left_sweep(tensors, epsrel, advance)
    return tensors


def _canonical_copy(tensors, advance=None):
    """Return an exact canonical copy and relative singular values per cut."""
    result = [tensor.copy() for tensor in tensors]
    _right_canonicalize(result, advance)
    spectra = _left_sweep(result, 0.0, advance)
    return result, spectra


def _significant_pairs(spectrum_a, spectrum_b, epsrel):
    """Select products above a relative cutoff without a dense outer product.

    Spectra are descending and normalized to their largest singular value.
    Zero tolerance explicitly retains every pair, including numerical zeros.
    """
    if epsrel == 0.0:
        return (
            np.repeat(np.arange(len(spectrum_a)), len(spectrum_b)),
            np.tile(np.arange(len(spectrum_b)), len(spectrum_a)))
    first_indices = []
    second_indices = []
    for index, value in enumerate(spectrum_a):
        if value <= epsrel:
            break
        # Searching negative descending spectra gives strict product > epsrel.
        count = int(np.searchsorted(
            -spectrum_b, -(epsrel / value), side="left"))
        if count:
            first_indices.extend([index] * count)
            second_indices.extend(range(count))
    if not first_indices:
        # A zero influence functional still needs nonempty tensor bonds.
        first_indices = [0]
        second_indices = [0]
    return (
        np.asarray(first_indices, dtype=np.intp),
        np.asarray(second_indices, dtype=np.intp))


def _product(tensors_a, tensors_b, advance=None):
    """Exact Hadamard influence product with Kronecker-product bonds."""
    result = []
    for tensor_a, tensor_b in zip(tensors_a, tensors_b):
        left_a, right_a, physical = tensor_a.shape
        left_b, right_b, _ = tensor_b.shape
        result.append(np.einsum(
            "abq,cdq->acbdq", tensor_a, tensor_b, optimize=True
        ).reshape((left_a * left_b, right_a * right_b, physical)))
        if advance is not None:
            advance()
    return result


def _preselected_product(tensors_a, tensors_b, epsrel, advance=None):
    """Merge only significant canonical bond pairs, then recompress.

    Selection occurs before allocating each product tensor. Its memory use
    is proportional to the selected adjacent bond dimensions, rather than
    to the full product of four input bond dimensions. The selection is an
    additional approximation for nonzero tolerance.
    """
    canonical_a, spectra_a = _canonical_copy(tensors_a, advance)
    canonical_b, spectra_b = _canonical_copy(tensors_b, advance)
    boundary = (np.array([0], dtype=np.intp),
                np.array([0], dtype=np.intp))
    pairs = [boundary]
    pairs.extend(
        _significant_pairs(spectrum_a, spectrum_b, epsrel)
        for spectrum_a, spectrum_b in zip(spectra_a, spectra_b))
    pairs.append(boundary)
    result = []
    for site, (tensor_a, tensor_b) in enumerate(
            zip(canonical_a, canonical_b)):
        left_a, left_b = pairs[site]
        right_a, right_b = pairs[site + 1]
        selected_a = tensor_a[left_a[:, None], right_a[None, :], :]
        selected_b = tensor_b[left_b[:, None], right_b[None, :], :]
        result.append(selected_a * selected_b)
        if advance is not None:
            advance()
    return _compress(result, epsrel, advance)


class AceBackend:
    """Build a compressed influence MPO for independent finite modes.

    Parameters
    ----------
    coupling_eigenvalues : array_like
        Eigenvalues of the common Hermitian system coupling operator.
    modes : iterable
        Mode objects exposing hamiltonian, coupling_operator, initial_state,
        gammas and lindblad_operators. Matrices use units with hbar = 1.
        Initial mode states must be normalized.
    dt : float
        Time interval for each environment propagator.
    num_steps : int
        Number of influence tensors to construct.
    epsrel : float
        Local SVD cutoff relative to the largest singular value. A value
        of zero disables singular-value truncation and pair preselection.
    method : {'preselection', 'sequential', 'tree'}
        Preselection uses a balanced hierarchy of pair merges, filtering
        canonical spectrum products before multiplication; sequential
        multiplies and compresses successive modes; tree uses a balanced
        hierarchy of exact pair products with compression at each node.

    Notes
    -----
    Each mode has conditional Hamiltonian H_a = H_mode + lambda_a B.
    For row-major density-vector flattening the conditional generator is

    -1j * (kron(H_a, I) - kron(I, H_b.T)) + dissipator.

    Every mode's exponential over one interval is exact up to numerical
    matrix exponentiation. The surrounding OQuPy dynamics calculation
    supplies the system/environment splitting. This representation assumes
    initially independent modes and one common system coupling basis.
    Noncommuting couplings and correlated initial modes require a different
    construction.

    Canonicalization distributes scalar factors across sites to avoid
    needless norm growth for long time grids. Neither nonzero SVD tolerance
    nor preselection guarantees positivity or a global observable error;
    results must be converged with respect to dt, epsrel and mode dimension.
    """

    def __init__(self, coupling_eigenvalues, modes, dt, num_steps,
                 epsrel, method_combination, method_svd, progress_type=None):
        self._eigenvalues = np.asarray(coupling_eigenvalues, dtype=float)
        self._modes = tuple(modes)
        self._dt = float(dt)
        self._num_steps = int(num_steps)
        self._epsrel = float(epsrel)
        self._method_combination = method_combination
        self._method_svd = method_svd
        self._bond_dimensions = None
        self.progress_type = progress_type
        if method_combination not in ('sequential', 'tree'):
            raise ValueError(
                'Unknown ACE combintation method:' + str(method_combination))
        if method_svd not in ('standard', 'preselection'):
            raise ValueError('Unknown SVD compression:' + str(method_svd))
        if self._num_steps < 0:
            raise ValueError("num_steps must be nonnegative.")
        if self._epsrel < 0.0 or self._epsrel >= 1.0:
            raise ValueError("epsrel must satisfy 0 <= epsrel < 1.")

    @property
    def bond_dimensions(self):
        """Bond dimensions of the last result, or None before computation."""
        if self._bond_dimensions is None:
            return None
        return list(self._bond_dimensions)

    def _mode_tensors(self, mode, advance=None):
        """Exponentiate conditional generators and close mode endpoints."""
        hamiltonian = np.asarray(mode.hamiltonian, dtype=complex)
        coupling = np.asarray(mode.coupling_operator, dtype=complex)
        initial = np.asarray(mode.initial_state, dtype=complex).reshape(-1)
        mode_dimension = hamiltonian.shape[0]
        identity = np.eye(mode_dimension, dtype=complex)
        environment_dimension = mode_dimension ** 2
        dissipator = np.zeros(
            (environment_dimension, environment_dimension), dtype=complex)
        for gamma, operator in zip(mode.gammas, mode.lindblad_operators):
            operator = np.asarray(operator, dtype=complex)
            product = operator.conj().T @ operator
            dissipator += gamma * (
                np.kron(operator, operator.conj())
                - 0.5 * np.kron(product, identity)
                - 0.5 * np.kron(identity, product.T))

        system_dimension = len(self._eigenvalues)
        kernel = np.empty((
            environment_dimension, environment_dimension,
            system_dimension ** 2), dtype=complex)
        conditional = [
            hamiltonian + eigenvalue * coupling
            for eigenvalue in self._eigenvalues]
        for ket, hamiltonian_ket in enumerate(conditional):
            for bra, hamiltonian_bra in enumerate(conditional):
                generator = -1.0j * (
                    np.kron(hamiltonian_ket, identity)
                    - np.kron(identity, hamiltonian_bra.T)) + dissipator
                kernel[:, :, ket * system_dimension + bra] = (
                    linalg.expm(self._dt * generator).T)
                if advance is not None:
                    advance()

        trace = identity.reshape(-1)
        if self._num_steps == 1:
            result = [np.einsum(
                "a,abq,b->q", initial, kernel, trace, optimize=True
            ).reshape((1, 1, system_dimension ** 2))]
        else:
            result = [np.einsum(
                "a,abq->bq", initial, kernel, optimize=True)[None, :, :]]
            result.extend(
                kernel.copy() for _ in range(self._num_steps - 2))
            result.append(np.einsum(
                "abq,b->aq", kernel, trace, optimize=True)[:, None, :])
        return _compress(result, self._epsrel, advance)

    def _merge(self, tensors_a, tensors_b, advance=None):
        if self._method_svd == "preselection":
            return _preselected_product(
                tensors_a, tensors_b, self._epsrel, advance)
        return _compress(
            _product(tensors_a, tensors_b, advance), self._epsrel, advance)

    def _compute(self, advance=None):
        """Return new influence tensors with unit boundary bonds."""
        if self._num_steps == 0:
            self._bond_dimensions = [1]
            return []
        if not self._modes:
            result = [np.ones(
                (1, 1, len(self._eigenvalues) ** 2), dtype=complex)
                for _ in range(self._num_steps)]
        elif self._method_combination in "tree":
            # Build subtrees on demand instead of retaining all mode MPOs.
            def combine(first, last):
                if last - first == 1:
                    return self._mode_tensors(self._modes[first], advance)
                middle = (first + last) // 2
                return self._merge(combine(first, middle),
                                   combine(middle, last), advance)

            result = combine(0, len(self._modes))
        else:
            mode_iterator = iter(self._modes)
            result = self._mode_tensors(next(mode_iterator), advance)
            for mode in mode_iterator:
                result = self._merge(
                    result, self._mode_tensors(mode, advance), advance)
        self._bond_dimensions = (
            [result[0].shape[0]] + [tensor.shape[1] for tensor in result])
        return result

    def compute(self, progress_type=None):
        """Build modes and merge/compress, reporting completed work units.

        Units count conditional exponentials, product sites and QR/SVD bonds,
        not elapsed-time fractions. A single matrix operation is indivisible.
        """
        selected = (
            self.progress_type if progress_type is None else progress_type)
        progress = get_progress(selected)
        count = len(self._modes)
        levels = ((count-1).bit_length() if count else 0)
        if self._method_combination == "sequential":
            levels = max(0, count-1)
        total = 1 + levels if self._num_steps else 1
        with progress(
                total, "--> ACE: build modes, then merge levels:") as report:
            if self._num_steps == 0:
                self._bond_dimensions = [1]
                report.update(1)
                return []
            if not count:
                result = [np.ones(
                    (1, 1, len(self._eigenvalues)**2), dtype=complex)
                    for _ in range(self._num_steps)]
                report.update(1)
            else:
                # Preserve the original balanced-tree grouping, including
                # non-power-of-two mode counts, while evaluating by level.
                nodes = {}
                schedule = {}
                def plan(first, last):
                    if last-first == 1:
                        return 0
                    middle = (first+last)//2
                    height = 1+max(plan(first, middle), plan(middle, last))
                    schedule.setdefault(height, []).append(
                        (first, middle, last))
                    return height
                if self._method_combination != "sequential":
                    plan(0, count)
                for index, mode in enumerate(self._modes):
                    nodes[index, index+1] = self._mode_tensors(mode)
                report.update(1)
                if self._method_combination == "sequential":
                    result = nodes.pop((0, 1))
                    for index in range(1, count):
                        result = self._merge(
                            result, nodes.pop((index, index+1)))
                        report.update(index+1)
                else:
                    for height in sorted(schedule):
                        for first, middle, last in schedule[height]:
                            left = nodes.pop((first, middle))
                            right = nodes.pop((middle, last))
                            nodes[first, last] = self._merge(left, right)
                        report.update(height+1)
                    result = nodes[0, count]
            self._bond_dimensions = (
                [result[0].shape[0]] + [tensor.shape[1] for tensor in result])
            return result

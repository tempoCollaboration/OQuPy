# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Automatic compression of environments (ACE) and legacy tensor conversion.

The finite modes share one Hermitian system coupling operator. Their initial
state is a product state, independent of the system. Hamiltonians and mode
couplings use frequency units (hbar=1), as in the rest of OQuPy.
"""

import os
from numbers import Integral

import numpy as np

from oqupy.base_api import BaseAPIClass
from oqupy.ace_bath import AceMode
from oqupy.backends.ace_backend import AceBackend
from oqupy.operators import left_right_super
from oqupy.process_tensor import FileProcessTensor, SimpleProcessTensor
from oqupy.util import get_progress


def _finite_real(value, name):
    """Convert a finite real scalar, rejecting booleans and complex values."""
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _basis_transforms(unitary, dimension):
    """OQuPy stores physical transforms with input/output axes transposed."""
    if unitary is None:
        return None, None
    unitary = np.array(unitary, dtype=complex, copy=True)
    if unitary.shape != (dimension, dimension) or not np.all(
            np.isfinite(unitary)):
        raise ValueError("unitary_transform has an invalid shape or entries.")
    if not np.allclose(unitary.conj().T @ unitary, np.eye(dimension),
                       rtol=1e-10, atol=1e-12):
        raise ValueError("unitary_transform must be unitary.")
    if np.array_equal(unitary, np.eye(dimension)):
        return None, None
    return (left_right_super(unitary.conj().T, unitary).T,
            left_right_super(unitary, unitary.conj().T).T)


class AceParameters(BaseAPIClass):
    """Parameters for finite-mode ACE.

    Parameters
    ----------
    dt : float
        Strictly positive time step.
    epsrel : float
        Singular-value cutoff relative to the largest singular value at each
        bond, in [0, 1). Zero disables truncation; it can be very expensive.
        This is not a global error bound or PT-TEMPO's discarded-norm tolerance.
    method_combination : str
        'sequential' : sequentially combine the modes together.
        'tree' : balanced tree combination.
    method_svd : str
        'standard' : standard SVD compression.
        'preselection' : preselection technique developed by Mortiz Cygorek.
    name, description : str, optional
        Standard OQuPy metadata.
    """

    def __init__(self, dt, epsrel=1e-6, method_combination='sequential',
                 method_svd="preselection", name=None, description=None):
        super().__init__(name, description)
        self._dt = _finite_real(dt, "dt")
        self._epsrel = _finite_real(epsrel, "epsrel")
        if self._dt <= 0:
            raise ValueError("dt must be strictly positive.")
        if not 0 <= self._epsrel < 1:
            raise ValueError("epsrel must satisfy 0 <= epsrel < 1.")
        if method_combination not in ('sequential', 'tree'):
            raise ValueError('method must be sequential or tree')
        if method_svd not in ('standard', 'preselection'):
            raise ValueError('svd must be standard or preselection')
        self._method_combination = method_combination
        self._method_svd = method_svd

    @property
    def dt(self):
        """Time step."""
        return self._dt

    @property
    def epsrel(self):
        """Relative singular-value threshold."""
        return self._epsrel

    @property
    def method_combination(self):
        """Mode combination method."""
        return self._method_combination

    @property
    def method_svd(self):
        """Mode SVD method."""
        return self._method_svd


class Ace(BaseAPIClass):
    """Construct a reusable OQuPy process tensor from independent finite modes.

    Parameters
    ----------
    coupling_operator : ndarray
        Hermitian system operator S in H = H_system + sum(H_k + S B_k).
        A common eigenbasis is found internally; pass system Hamiltonians,
        initial states and observables to OQuPy in the original system basis.
    modes : iterable of AceMode
        Independent modes. An empty iterable gives the identity environment.
    start_time, end_time : float
        Interval containing a positive integer number of time steps.
    parameters : AceParameters
        Time step, compression threshold and merge method.
    process_tensor_file : str, path-like or bool, optional
        None/False returns SimpleProcessTensor; a path returns a writable
        FileProcessTensor; True creates a temporary FileProcessTensor.
        File output persists the result but computation still uses memory.
    overwrite : bool
        Allow replacing an existing process tensor file.
    name, description : str, optional
        Standard OQuPy metadata.

    Notes
    -----
    Only environment dynamics is encoded here. OQuPy's compute_dynamics
    inserts the two system half steps. The returned tensor does not store
    start_time: pass it separately to compute_dynamics if it is nonzero.
    """

    def __init__(self, coupling_operator, modes, start_time, end_time,
                 parameters, process_tensor_file=None, overwrite=False,
                 name=None, description=None):
        super().__init__(name, description)
        if not isinstance(parameters, AceParameters):
            raise TypeError("parameters must be an AceParameters object.")
        coupling = np.array(coupling_operator, dtype=complex, copy=True)
        if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1] \
                or coupling.shape[0] == 0 or not np.all(np.isfinite(coupling)):
            raise ValueError(
                "coupling_operator must be a finite square matrix.")
        if not np.allclose(coupling, coupling.conj().T,
                           rtol=1e-10, atol=1e-12):
            raise ValueError("coupling_operator must be Hermitian.")
        self._modes = tuple(modes)
        if any(not isinstance(mode, AceMode) for mode in self._modes):
            raise TypeError("Each mode must be an AceMode object.")
        self._start_time = _finite_real(start_time, "start_time")
        self._end_time = _finite_real(end_time, "end_time")
        steps = (self._end_time - self._start_time) / parameters.dt
        if not np.isfinite(steps) or steps <= 0:
            raise ValueError(
                "The time interval must contain at least one step.")
        self._num_steps = int(round(steps))
        if self._num_steps < 1 or not np.isclose(
                steps, self._num_steps, rtol=0, atol=1e-9):
            raise ValueError(
                "(end_time - start_time) must be a multiple of dt.")
        if process_tensor_file is not None and not isinstance(
                process_tensor_file, (str, os.PathLike, bool)):
            raise TypeError("process_tensor_file must be a path or bool.")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool.")
        self._process_tensor_file = process_tensor_file
        self._overwrite = overwrite
        self._parameters = parameters
        self._dimension = coupling.shape[0]
        self._eigenvalues, self._unitary = np.linalg.eigh(coupling)
        self._process_tensor = None
        self._backend_instance = AceBackend(
            self._eigenvalues, self._modes, parameters.dt, self._num_steps,
            parameters.epsrel, parameters.method_combination,
            parameters.method_svd)

    @property
    def dimension(self):
        """System Hilbert-space dimension."""
        return self._dimension

    @property
    def num_steps(self):
        """Number of process tensor sites."""
        return self._num_steps

    def compute(self, progress_type=None):
        """Compute once; subsequent calls reuse the completed process tensor."""
        if self._process_tensor is not None:
            return
        tensors = self._backend_instance.compute(progress_type=progress_type)
        with get_progress("silent")(1) as report:
            transform_in, transform_out = _basis_transforms(
                self._unitary, self._dimension)
            options = {
                "hilbert_space_dimension": self._dimension,
                "dt": self._parameters.dt, "transform_in": transform_in,
                "transform_out": transform_out, "name": self.name,
                "description": self.description}
            destination = self._process_tensor_file
            if destination is None or destination is False:
                process_tensor = SimpleProcessTensor(**options)
            else:
                process_tensor = FileProcessTensor(
                    mode="overwrite" if self._overwrite else "write",
                    filename=None if destination is True else os.fspath(
                        destination), **options)
            try:
                for step, tensor in enumerate(tensors):
                    process_tensor.set_mpo_tensor(step, tensor)
                process_tensor.compute_caps()
            except Exception:
                if isinstance(process_tensor, FileProcessTensor):
                    process_tensor.close()
                raise
            self._process_tensor = process_tensor
            report.update(1)

    def get_process_tensor(self, progress_type=None):
        """Compute if needed and return the native process tensor."""
        self.compute(progress_type=progress_type)
        return self._process_tensor


def ace_compute(coupling_operator, modes, start_time, end_time, parameters,
                process_tensor_file=None, overwrite=False, progress_type=None,
                name=None, description=None):
    """Compute a native process tensor; see Ace for the parameters."""
    calculation = Ace(
        coupling_operator, modes, start_time, end_time, parameters,
        process_tensor_file, overwrite, name, description)
    return calculation.get_process_tensor(progress_type=progress_type)


def import_ace_process_tensor(tensors, hilbert_space_dimension, dt,
                              unitary_transform=None, name=None,
                              description=None):
    """Convert the supplied research code's Q_MPO list into OQuPy format.

    Parameters
    ----------
    tensors : iterable of ndarray
        Chronological legacy tensors with axes
        (system_output, system_input, future_bond, past_bond).
        The diagonal form has shape (d**2, 1, future_bond, past_bond).
        The full form has shape (d**2, d**2, future_bond, past_bond).
        Initial bath states and final traces must already be contracted.
    hilbert_space_dimension : int
        System Hilbert-space dimension d, not the Liouville dimension d**2.
    dt : float
        Positive time step in the same units used by the original calculation.
    unitary_transform : ndarray, optional
        Columns are the coupling eigenvectors in the original system basis,
        if the original tensors were calculated in that eigenbasis.
    name, description : str, optional
        Standard OQuPy metadata.

    Returns
    -------
    SimpleProcessTensor
        A native tensor with OQuPy trace caps. Conversion copies the arrays
        and does not recompress or correct the original calculation.
    """
    if isinstance(hilbert_space_dimension, bool) or not isinstance(
            hilbert_space_dimension, Integral) or hilbert_space_dimension < 1:
        raise ValueError("hilbert_space_dimension must be a positive integer.")
    dimension = int(hilbert_space_dimension)
    dt = _finite_real(dt, "dt")
    if dt <= 0:
        raise ValueError("dt must be strictly positive.")
    transform_in, transform_out = _basis_transforms(
        unitary_transform, dimension)
    converted = []
    previous_bond = 1
    for step, tensor in enumerate(tensors):
        array = np.array(tensor, dtype=complex, copy=True)
        if array.ndim != 4 or array.shape[0] != dimension**2 \
                or array.shape[1] not in (1, dimension**2) \
                or min(array.shape) < 1 or not np.all(np.isfinite(array)):
            raise ValueError(
                f"Invalid legacy MPO shape or values at step {step}.")
        if array.shape[3] != previous_bond:
            raise ValueError(f"Legacy MPO bonds do not match at step {step}.")
        previous_bond = array.shape[2]
        if array.shape[1] == 1:
            converted.append(array[:, 0, :, :].transpose(2, 1, 0))
        else:
            converted.append(array.transpose(3, 2, 1, 0))
    if not converted or previous_bond != 1:
        raise ValueError(
            "Legacy MPO must be nonempty with unit boundary bonds.")
    process_tensor = SimpleProcessTensor(
        hilbert_space_dimension=dimension, dt=dt, transform_in=transform_in,
        transform_out=transform_out, name=name, description=description)
    for step, tensor in enumerate(converted):
        process_tensor.set_mpo_tensor(step, tensor)
    process_tensor.compute_caps()
    return process_tensor

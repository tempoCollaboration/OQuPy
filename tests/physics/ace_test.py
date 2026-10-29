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
"""Scientific regressions for ACE, independent of its tensor construction.

The reference solver constructs the complete system-environment Liouvillian
directly in the original Hilbert-space basis. It uses the same symmetric
system/environment splitting as OQuPy, so these comparisons isolate tensor,
basis, vectorization, cap, and compression errors from time-discretization
error. A separate convergence test checks the latter.

Run with either pytest or:
    python -m unittest tests.physics.ace_test -v
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

import oqupy
from oqupy.process_tensor import BaseProcessTensor, import_process_tensor


SX = np.array([[0., 1.], [1., 0.]], dtype=complex)
SY = np.array([[0., -1.j], [1.j, 0.]], dtype=complex)
SZ = np.diag([1., -1.]).astype(complex)


def _pure_state(vector):
    vector = np.asarray(vector, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    return np.outer(vector, vector.conj())


def _kron_all(operators):
    result = np.ones((1, 1), dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def _embed(operator, position, dimensions):
    operators = [np.eye(dimension) for dimension in dimensions]
    operators[position] = operator
    return _kron_all(operators)


def _mode(hamiltonian, coupling_operator, initial_state,
          gammas=None, lindblad_operators=None):
    """Keep the independent reference inputs separate from ACE internals."""
    return dict(
        hamiltonian=np.asarray(hamiltonian, dtype=complex),
        coupling_operator=np.asarray(coupling_operator, dtype=complex),
        initial_state=np.asarray(initial_state, dtype=complex),
        gammas=[] if gammas is None else gammas,
        lindblad_operators=[] if lindblad_operators is None
        else lindblad_operators)


def _joint_generator(coupling_operator, modes):
    """Construct a row-major joint Liouvillian using elementary Kroneckers."""
    system_dimension = coupling_operator.shape[0]
    dimensions = [mode["hamiltonian"].shape[0] for mode in modes]
    environment_dimension = int(np.prod(dimensions))
    joint_dimension = system_dimension * environment_dimension
    hamiltonian = np.zeros((joint_dimension, joint_dimension), dtype=complex)
    collapse_operators = []
    for index, mode in enumerate(modes):
        mode_hamiltonian = _embed(mode["hamiltonian"], index, dimensions)
        mode_coupling = _embed(mode["coupling_operator"], index, dimensions)
        hamiltonian += np.kron(np.eye(system_dimension), mode_hamiltonian)
        hamiltonian += np.kron(coupling_operator, mode_coupling)
        for rate, operator in zip(mode["gammas"], mode["lindblad_operators"]):
            collapse_operators.append((
                rate, np.kron(np.eye(system_dimension),
                              _embed(operator, index, dimensions))))
    identity = np.eye(joint_dimension)
    generator = -1.j * (
        np.kron(hamiltonian, identity)
        - np.kron(identity, hamiltonian.T))
    for rate, operator in collapse_operators:
        product = operator.conj().T @ operator
        generator += rate * (
            np.kron(operator, operator.conj())
            - 0.5 * np.kron(product, identity)
            - 0.5 * np.kron(identity, product.T))
    return generator, hamiltonian, environment_dimension


def _dense_dynamics(system_hamiltonian, coupling_operator, modes,
                    initial_state, dt, num_steps):
    generator, _, environment_dimension = _joint_generator(
        coupling_operator, modes)
    dimension = initial_state.shape[0]
    joint_state = np.kron(
        initial_state, _kron_all([mode["initial_state"] for mode in modes]))
    environment_step = expm(dt * generator)
    system_half = np.kron(
        expm(-0.5j * dt * system_hamiltonian),
        np.eye(environment_dimension))

    def reduce_state(state):
        return np.einsum(
            "aibi->ab",
            state.reshape(dimension, environment_dimension,
                          dimension, environment_dimension))

    states = [reduce_state(joint_state)]
    for _ in range(num_steps):
        joint_state = system_half @ joint_state @ system_half.conj().T
        joint_state = (environment_step @ joint_state.ravel()).reshape(
            joint_state.shape)
        joint_state = system_half @ joint_state @ system_half.conj().T
        states.append(reduce_state(joint_state))
    return np.asarray(states)


class AcePhysicsTest(unittest.TestCase):
    """Compare complete density matrices, including complex coherences."""

    def setUp(self):
        self.system_hamiltonian = 0.43 * SX + 0.17 * SY - 0.11 * SZ
        self.coupling = 0.37 * SX + 0.23 * SY + 0.41 * SZ
        self.initial_state = _pure_state([1., 0.4 + 0.7j])
        self.modes = [
            _mode(0.61 * SZ + 0.13 * SY, 0.27 * SX - 0.09 * SY,
                  _pure_state([0.6, 0.8j])),
            _mode(0.29 * SZ - 0.17 * SX, 0.19 * SY + 0.07 * SZ,
                  np.array([[0.65, 0.08j], [-0.08j, 0.35]])),
        ]

    def _make_process(self, modes=None, coupling=None, dt=0.12, steps=3,
                      method_combination="sequential", method_svd = 'preselection', epsrel=0., **kwargs):
        modes = self.modes if modes is None else modes
        coupling = self.coupling if coupling is None else coupling
        process = oqupy.ace_compute(
            coupling_operator=coupling,
            modes=[oqupy.AceMode(**mode) for mode in modes],
            start_time=0., end_time=steps * dt,
            parameters=oqupy.AceParameters(
                dt=dt, epsrel=epsrel, method_combination=method_combination,
                method_svd = method_svd),
            progress_type="silent", **kwargs)
        self.assertIsInstance(process, BaseProcessTensor)
        self.assertEqual(len(process), steps)
        self.assertAlmostEqual(process.dt, dt)
        return process

    def _compare_dense(self, process, modes=None, coupling=None,
                       hamiltonian=None, initial_state=None, atol=2.e-11):
        modes = self.modes if modes is None else modes
        coupling = self.coupling if coupling is None else coupling
        hamiltonian = self.system_hamiltonian if hamiltonian is None \
            else hamiltonian
        initial_state = self.initial_state if initial_state is None \
            else initial_state
        states = oqupy.compute_dynamics(
            oqupy.System(hamiltonian), initial_state=initial_state,
            process_tensor=process, progress_type="silent").states
        expected = _dense_dynamics(
            hamiltonian, coupling, modes, initial_state,
            process.dt, len(process))
        assert_allclose(states, expected, atol=atol, rtol=atol)
        assert_allclose(np.trace(states, axis1=1, axis2=2),
                        np.ones(len(states)), atol=atol, rtol=atol)
        assert_allclose(states, states.conj().transpose(0, 2, 1),
                        atol=atol, rtol=atol)
        return states

    def test_all_merge_methods_with_complex_coupling_and_odd_mode_count(self):
        modes = self.modes + [
            _mode(0.38 * SZ + 0.08 * SY, 0.12 * SX + 0.03 * SZ,
                  _pure_state([0.3j, 1.]))]
        for method in ("sequential", "tree"):
            with self.subTest(method=method):
                process = self._make_process(modes=modes, method_combination=method)
                self._compare_dense(process, modes=modes)
                
    def test_all_svd_with_complex_coupling_and_odd_mode_count(self):
        modes = self.modes + [
            _mode(0.38 * SZ + 0.08 * SY, 0.12 * SX + 0.03 * SZ,
                  _pure_state([0.3j, 1.]))]
        for method in ("standard", "preselection"):
            with self.subTest(method=method):
                process = self._make_process(modes=modes, method_svd=method)
                self._compare_dense(process, modes=modes)

    def test_qutrit_complex_basis_and_two_modes(self):
        coupling = np.array([
            [0.5, 0.2 + 0.1j, -0.07j],
            [0.2 - 0.1j, -0.3, 0.13],
            [0.07j, 0.13, 0.8]])
        hamiltonian = np.array([
            [0., 0.4j, 0.17], [-0.4j, 0.7, 0.12j],
            [0.17, -0.12j, 1.1]])
        initial_state = _pure_state([1., 0.4j, 0.3 - 0.2j])
        process = self._make_process(coupling=coupling)
        self.assertEqual(process.hilbert_space_dimension, 3)
        self._compare_dense(process, coupling=coupling,
                            hamiltonian=hamiltonian,
                            initial_state=initial_state)

    def test_lindblad_environment_matches_joint_master_equation(self):
        lowering = np.array([[0., 1.], [0., 0.]])
        modes = [
            _mode(0.5 * SZ + 0.1 * SY, 0.31 * SX,
                  _pure_state([1., 0.5j]),
                  gammas=[0.39, 0.13],
                  lindblad_operators=[lowering, SZ])]
        process = self._make_process(modes=modes, steps=4)
        self._compare_dense(process, modes=modes)

    def test_harmonic_finite_temperature_matches_explicit_mode(self):
        omega, strength, temperature, levels = 0.9, 0.24, 0.67, 3
        annihilation = np.diag(np.sqrt(np.arange(1, levels)), 1)
        hamiltonian = omega * np.diag(np.arange(levels))
        coupling = strength * (annihilation + annihilation.T)
        probabilities = np.exp(-omega * np.arange(levels) / temperature)
        initial_state = np.diag(probabilities / probabilities.sum())
        modes = [_mode(hamiltonian, coupling, initial_state)]
        process = oqupy.ace_compute(
            self.coupling,
            [oqupy.AceMode.harmonic(
                omega, strength, levels, temperature=temperature)],
            0., 0.36, oqupy.AceParameters(0.12, epsrel=0.),
            progress_type="silent")
        self._compare_dense(process, modes=modes)

    def test_thermal_mode_uses_hamiltonian_eigenbasis(self):
        hamiltonian = 0.41 * SY + 0.27 * SZ
        coupling = 0.23 * SX
        for temperature in (0., 0.61):
            with self.subTest(temperature=temperature):
                if temperature == 0.:
                    _, vectors = np.linalg.eigh(hamiltonian)
                    initial_state = np.outer(
                        vectors[:, 0], vectors[:, 0].conj())
                else:
                    initial_state = expm(-hamiltonian / temperature)
                    initial_state /= np.trace(initial_state)
                modes = [_mode(hamiltonian, coupling, initial_state)]
                process = oqupy.ace_compute(
                    self.coupling,
                    [oqupy.AceMode.thermal(
                        hamiltonian, coupling, temperature)],
                    0., 0.36, oqupy.AceParameters(0.12, epsrel=0.),
                    progress_type="silent")
                self._compare_dense(process, modes=modes)

    def test_single_step_initial_and_final_caps(self):
        process = self._make_process(steps=1)
        self._compare_dense(process)

    def test_empty_environment_is_identity_for_all_methods(self):
        for method in ("sequential", "tree"):
            with self.subTest(method=method):
                process = self._make_process(modes=[], method_combination=method)
                self._compare_dense(process, modes=[])
                assert_allclose(process.get_bond_dimensions(), 1)

    def test_process_reuse_with_different_systems_and_states(self):
        process = self._make_process()
        before = [process.get_mpo_tensor(step).copy()
                  for step in range(len(process))]
        self._compare_dense(process)
        self._compare_dense(
            process, hamiltonian=-0.19 * SX + 0.34 * SZ,
            initial_state=_pure_state([0.2 - 0.3j, 1.]))
        for step, original in enumerate(before):
            assert_allclose(process.get_mpo_tensor(step), original,
                            atol=0., rtol=0.)

    # def test_compressed_methods_retain_dense_dynamics(self):
    #     for method in ("sequential", "tree", "preselection"):
    #         with self.subTest(method=method):
    #             process = self._make_process(
    #                 method=method, epsrel=1.e-9, steps=5)
    #             self._compare_dense(process, atol=2.e-7)

    def test_file_export_and_import_preserve_basis_and_dynamics(self):
        process = self._make_process(name="ACE regression")
        expected = self._compare_dense(process)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "ace_process.hdf5")
            process.export(filename)
            restored = import_process_tensor(filename)
            try:
                self.assertEqual(restored.name, "ACE regression")
                assert_allclose(self._compare_dense(restored), expected,
                                atol=2.e-12, rtol=2.e-12)
            finally:
                restored.close()

    def test_direct_file_output_and_builder_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "direct_ace.hdf5")
            builder = oqupy.Ace(
                self.coupling,
                [oqupy.AceMode(**mode) for mode in self.modes],
                0., 0.36, oqupy.AceParameters(0.12, epsrel=0.),
                process_tensor_file=filename)
            process = builder.get_process_tensor(progress_type="silent")
            try:
                self.assertIs(
                    builder.get_process_tensor(progress_type="silent"),
                    process)
                self._compare_dense(process)
            finally:
                if hasattr(process, "close"):
                    process.close()
            self.assertTrue(Path(filename).is_file())

    def test_second_order_time_step_convergence(self):
        modes = self.modes[:1]
        _, interaction_hamiltonian, environment_dimension = _joint_generator(
            self.coupling, modes)
        total_hamiltonian = interaction_hamiltonian + np.kron(
            self.system_hamiltonian, np.eye(environment_dimension))
        joint_initial = np.kron(self.initial_state, modes[0]["initial_state"])
        final_time = 0.6
        unitary = expm(-1.j * final_time * total_hamiltonian)
        exact_joint = unitary @ joint_initial @ unitary.conj().T
        exact_final = np.einsum(
            "aibi->ab", exact_joint.reshape(2, environment_dimension,
                                           2, environment_dimension))
        errors = []
        for dt in (0.15, 0.075):
            process = self._make_process(
                modes=modes, dt=dt, steps=round(final_time / dt))
            states = self._compare_dense(process, modes=modes)
            errors.append(np.linalg.norm(states[-1] - exact_final))
        self.assertGreater(errors[0], 1.e-8)
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertLess(errors[0] / errors[1], 4.2)

    def test_legacy_full_tensor_axis_order(self):
        unitary = expm(-0.13j * (SX + 0.7 * SY))
        channel = np.kron(unitary, unitary.conj())
        raw = [channel[:, :, None, None].copy() for _ in range(3)]
        process = oqupy.import_ace_process_tensor(
            raw, hilbert_space_dimension=2, dt=0.1)
        states = oqupy.compute_dynamics(
            oqupy.System(np.zeros((2, 2))),
            initial_state=self.initial_state,
            process_tensor=process, progress_type="silent").states
        expected = [self.initial_state]
        for _ in range(3):
            expected.append(unitary @ expected[-1] @ unitary.conj().T)
        assert_allclose(states, expected, atol=2.e-12, rtol=2.e-12)

    def test_legacy_diagonal_tensor_and_complex_basis_transform(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.coupling)
        unitary_diagonal = np.exp(-0.23j * eigenvalues)
        diagonal_channel = np.outer(
            unitary_diagonal, unitary_diagonal.conj()).ravel()
        raw = [diagonal_channel[:, None, None, None].copy()
               for _ in range(3)]
        process = oqupy.import_ace_process_tensor(
            raw, hilbert_space_dimension=2, dt=0.1,
            unitary_transform=eigenvectors)
        unitary = eigenvectors @ np.diag(unitary_diagonal) \
            @ eigenvectors.conj().T
        states = oqupy.compute_dynamics(
            oqupy.System(np.zeros((2, 2))),
            initial_state=self.initial_state,
            process_tensor=process, progress_type="silent").states
        expected = [self.initial_state]
        for _ in range(3):
            expected.append(unitary @ expected[-1] @ unitary.conj().T)
        assert_allclose(states, expected, atol=2.e-12, rtol=2.e-12)


class AceValidationTest(unittest.TestCase):
    """Reject malformed models before any tensor construction."""

    def test_invalid_time_steps_and_tolerances(self):
        for dt in (0., -0.1, np.inf, np.nan):
            with self.subTest(dt=dt):
                with self.assertRaises((ValueError, TypeError)):
                    oqupy.AceParameters(dt=dt)
        for epsrel in (-0.01, np.inf, np.nan):
            with self.subTest(epsrel=epsrel):
                with self.assertRaises((ValueError, TypeError)):
                    oqupy.AceParameters(dt=0.1, epsrel=epsrel)
        with self.assertRaises((ValueError, TypeError)):
            oqupy.AceParameters(dt=0.1, method="unknown")

    def test_invalid_mode_matrices_and_density_matrices(self):
        valid = dict(hamiltonian=SZ, coupling_operator=SX,
                     initial_state=np.eye(2) / 2)
        for field, value in (
                ("hamiltonian", np.ones((2, 3))),
                ("hamiltonian", np.array([[0., 1.], [0., 0.]])),
                ("coupling_operator", np.eye(3)),
                ("initial_state", np.diag([1.2, -0.2])),
                ("initial_state", np.diag([0.2, 0.2])),
                ("initial_state", np.array([[0.5, 0.1], [0., 0.5]]))):
            with self.subTest(field=field, value=value):
                arguments = dict(valid)
                arguments[field] = value
                with self.assertRaises((ValueError, TypeError)):
                    oqupy.AceMode(**arguments)

    def test_invalid_time_intervals_and_system_coupling(self):
        parameters = oqupy.AceParameters(0.1)
        for start, end in ((0., 0.), (0.2, 0.1), (0., 0.25)):
            with self.subTest(start=start, end=end):
                with self.assertRaises((ValueError, TypeError)):
                    oqupy.Ace(SZ, [], start, end, parameters)
        with self.assertRaises((ValueError, TypeError)):
            oqupy.Ace(np.array([[0., 1.], [0., 0.]]), [],
                      0., 0.3, parameters)

    def test_legacy_inconsistent_bonds_are_rejected(self):
        tensors = [np.ones((4, 1, 2, 1)), np.ones((4, 1, 1, 3))]
        with self.assertRaises((ValueError, TypeError)):
            oqupy.import_ace_process_tensor(tensors, 2, 0.1)

    def test_invalid_lindblad_rates_or_operator_count(self):
        for rates, operators in (
                ([-0.1], [SX]), ([0.1, 0.2], [SX]),
                ([np.nan], [SX]), ([0.1], [np.eye(3)])):
            with self.subTest(rates=rates):
                with self.assertRaises((ValueError, TypeError)):
                    oqupy.AceMode(
                        SZ, SX, np.eye(2) / 2,
                        gammas=rates, lindblad_operators=operators)


if __name__ == "__main__":
    unittest.main()

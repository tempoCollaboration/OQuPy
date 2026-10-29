# Licensed under the Apache License, Version 2.0.
"""ACE model constructors, migration reference and existing OQuPy consumers."""
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

import oqupy
from tests.physics.ace_test import (
    SX, SY, SZ, _dense_dynamics, _joint_generator, _mode, _pure_state)


class AceModelsTest(unittest.TestCase):
    """Test physical helper conventions and integration with native consumers."""

    def test_nonzero_start_time_roundoff(self):
        process = oqupy.ace_compute(
            SZ, [], 0.2, 0.3, oqupy.AceParameters(0.1),
            progress_type="silent")
        dynamics = oqupy.compute_dynamics(
            oqupy.System(SX), np.diag([1., 0.]), start_time=0.2,
            process_tensor=process, progress_type="silent")
        assert_allclose(dynamics.times, [0.2, 0.3])
        self.assertEqual(len(process), 1)

    def test_original_mapping_reference_with_signed_modes(self):
        filename = Path(__file__).parents[1] / "data/ace_legacy_reference.npz"
        with np.load(filename) as data:
            modes = []
            for omega, strength, levels in zip(
                    data["omega"], data["strength"], data["levels"]):
                lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
                modes.append(oqupy.AceMode(
                    np.diag(omega * np.arange(levels)),
                    strength * (lowering + lowering.T),
                    np.diag([1.] + [0.] * (levels - 1))))
            process = oqupy.ace_compute(
                data["coupling"], modes, 0., float(data["dt"] * data["steps"]),
                oqupy.AceParameters(float(data["dt"]), 1e-10),
                progress_type="silent")
            states = oqupy.compute_dynamics(
                oqupy.System(data["hamiltonian"]), data["state"],
                process_tensor=process, progress_type="silent").states
            assert_allclose(states, data["expected"], atol=1e-8, rtol=1e-8)

    # def test_quadrature_integrates_known_polynomial(self):
    #     omega, coupling = oqupy.discretize_spectral_density(
    #         lambda value: 2 * value, 0., 3., 4, temperature = 0)
    #     assert_allclose(np.sum(coupling**2), 9., atol=1e-13)
    #     assert_allclose(np.sum(coupling**2 * omega), 18., atol=1e-13)
    #     self.assertTrue(np.all((omega > 0) & (omega < 3)))
    #     with self.assertRaises(ValueError):
    #         oqupy.discretize_spectral_density(lambda _: -1., 0., 3., 4)

    # def test_quadrature_accepts_native_spectral_density(self):
    #     density = oqupy.CustomSD(
    #         lambda value: value**2, cutoff=3., cutoff_type="hard")
    #     _, coupling = oqupy.discretize_spectral_density(
    #         density.spectral_density, 0., 3., 4, temperature = 0)
    #     assert_allclose(np.sum(coupling**2), 9., atol=1e-13)

    def test_model_arrays_are_isolated_from_caller_mutations(self):
        hamiltonian, coupling, state = SZ.copy(), SX.copy(), np.eye(2) / 2
        mode = oqupy.AceMode(
            hamiltonian, coupling, state, [0.2], [SY.copy()])
        hamiltonian[:] = 0
        coupling[:] = 0
        state[:] = 0
        mode.hamiltonian[:] = 0
        mode.initial_state[:] = 0
        mode.lindblad_operators[0][:] = 0
        assert_allclose(mode.hamiltonian, SZ)
        assert_allclose(mode.coupling_operator, SX)
        assert_allclose(mode.initial_state, np.eye(2) / 2)
        assert_allclose(mode.lindblad_operators[0], SY)

    def test_zero_temperature_degenerate_ground_state(self):
        mode = oqupy.AceMode.thermal(
            np.diag([-2., -2., 3.]),
            np.eye(3),
            0.,
        )
    
        assert_allclose(
            mode.initial_state,
            np.diag([0.5, 0.5, 0.]),
        )

    # def test_anharmonic_energies_and_harmonic_limit(self):
    #     mode = oqupy.AceMode.anharmonic(0.7, 0.2, 3, 0.4, 0.6)
    #     assert_allclose(np.diag(mode.hamiltonian), [0., 0.7, 1.8])
    #     harmonic = oqupy.AceMode.harmonic(0.7, 0.2, 3, 0.6)
    #     limit = oqupy.AceMode.anharmonic(0.7, 0.2, 3, 0., 0.6)
    #     assert_allclose(limit.hamiltonian, harmonic.hamiltonian)
    #     assert_allclose(limit.initial_state, harmonic.initial_state)

    # def test_morse_bound_states_and_centering(self):
    #     mode = oqupy.AceMode.morse(0.7, 0.2, 3, depth=2.1,
    #                                temperature=0.6)
    #     assert_allclose(np.diag(mode.hamiltonian), [0., 0.7, 0.9625])
    #     assert_allclose(
    #         np.trace(mode.coupling_operator @ mode.initial_state), 0.,
    #         atol=1e-14)
    #     self.assertGreater(abs(mode.coupling_operator[0, 1]), 0)
    #     with self.assertRaises(ValueError):
    #         oqupy.AceMode.morse(0.7, 0.2, 4, depth=2.1)

    # def test_pseudomode_transition_resolved_thermal_stationarity(self):
    #     mode = oqupy.AceMode.harmonic_pseudomode(0.8, 0.2, 3, 0.3, 0.7)
    #     self.assertEqual(len(mode.gammas), 4)
    #     state = mode.initial_state
    #     derivative = np.zeros_like(state)
    #     for rate, jump in zip(mode.gammas, mode.lindblad_operators):
    #         product = jump.conj().T @ jump
    #         derivative += rate * (
    #             jump @ state @ jump.conj().T
    #             - (product @ state + state @ product) / 2)
    #     assert_allclose(derivative, 0., atol=1e-14)
    #     self.assertEqual(np.count_nonzero(mode.lindblad_operators[0]), 1)
    #     self.assertEqual(np.count_nonzero(mode.lindblad_operators[2]), 1)

    def test_complex_collapse_operator(self):
        mode = _mode(0.4 * SZ, 0.2 * SY, np.eye(2) / 2,
                     [0.3], [np.array([[0.2j, 1.], [0.1j, -0.2]])])
        process = oqupy.ace_compute(
            SX, [oqupy.AceMode(**mode)], 0, 0.4,
            oqupy.AceParameters(0.1, 0), progress_type="silent")
        initial = _pure_state([1., 1.j])
        actual = oqupy.compute_dynamics(
            oqupy.System(0.3 * SZ), initial, process_tensor=process,
            progress_type="silent").states
        expected = _dense_dynamics(0.3 * SZ, SX, [mode], initial, 0.1, 4)
        assert_allclose(actual, expected, atol=1e-12)

    def test_multiple_native_process_tensors_match_combined_modes(self):
        modes = [oqupy.AceMode.harmonic(0.7, 0.2, 2, 0, True),
                 oqupy.AceMode.harmonic(1.1, 0.1, 3, 0, True)]
        parameters = oqupy.AceParameters(0.1, 0)
        separate = [oqupy.ace_compute(
            SY, [mode], 0, 0.4, parameters, progress_type="silent")
            for mode in modes]
        combined = oqupy.ace_compute(
            SY, modes, 0, 0.4, parameters, progress_type="silent")
        arguments = dict(system=oqupy.System(0.3 * SX),
                         initial_state=np.diag([1., 0.]),
                         progress_type="silent")
        actual = oqupy.compute_dynamics(
            process_tensor=separate, **arguments).states
        expected = oqupy.compute_dynamics(
            process_tensor=combined, **arguments).states
        assert_allclose(actual, expected, atol=2e-12)

    def test_control_pulse_with_environment_memory(self):
        mode = _mode(0.6 * SZ, 0.3 * SX, _pure_state([1., 0.4j]))
        dt, steps = 0.1, 4
        coupling, hamiltonian = 0.7 * SY, 0.2 * SZ
        initial = _pure_state([1., 0.5])
        process = oqupy.ace_compute(
            coupling, [oqupy.AceMode(**mode)], 0, dt * steps,
            oqupy.AceParameters(dt, 0), progress_type="silent")
        pulse = expm(-0.3j * SX)
        control = oqupy.Control(2)
        control.add_single(2, np.kron(pulse, pulse.conj()))
        actual = oqupy.compute_dynamics(
            oqupy.System(hamiltonian), initial, process_tensor=process,
            control=control, progress_type="silent").states
        generator, _, environment_dimension = _joint_generator(
            coupling, [mode])
        half = np.kron(expm(-0.5j * dt * hamiltonian),
                       np.eye(environment_dimension))
        step = expm(dt * generator)
        joint = np.kron(initial, mode["initial_state"])
        expected = []
        for index in range(steps + 1):
            if index == 2:
                joint_pulse = np.kron(pulse, np.eye(environment_dimension))
                joint = joint_pulse @ joint @ joint_pulse.conj().T
            expected.append(np.einsum(
                "aibi->ab", joint.reshape(2, environment_dimension,
                                         2, environment_dimension)))
            if index < steps:
                joint = half @ joint @ half.conj().T
                joint = (step @ joint.ravel()).reshape(joint.shape)
                joint = half @ joint @ half.conj().T
        assert_allclose(actual, expected, atol=2e-12)


if __name__ == "__main__":
    unittest.main()

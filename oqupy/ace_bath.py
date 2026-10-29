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
"""Finite environment modes for ACE, using angular frequency units (hbar=1).

Temperature means k_B T / hbar in the same units, not Kelvin. The interaction
is S (x) coupling_operator; S is supplied separately to the ACE calculation.
"""
import numbers

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gammaln


def _real_scalar(value, name, minimum=None, strict=False):
    """Validate a finite real scalar without accepting strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None:
        if result < minimum or (strict and result == minimum):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}.")
    return result


def _levels(n_levels):
    if isinstance(n_levels, (bool, np.bool_)) or not isinstance(
            n_levels, numbers.Integral):
        raise TypeError("n_levels must be an integer.")
    if n_levels < 1:
        raise ValueError("n_levels must be positive.")
    return int(n_levels)


def _matrix(value, name, dimension=None, hermitian=False):
    try:
        result = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix.") from exc
    if result.ndim != 2 or result.shape[0] == 0 or (
            result.shape[0] != result.shape[1]):
        raise ValueError(f"{name} must be a nonempty square matrix.")
    if dimension is not None and result.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if hermitian:
        if not np.allclose(result, result.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must be Hermitian.")
        result = (result + result.conj().T) / 2
    return result


def _gibbs_state(hamiltonian, temperature, thermalisation=False):
    """Stable Gibbs state; at zero temperature mix degenerate ground states."""
    temperature = _real_scalar(temperature, "temperature", minimum=0.0)
    energies, vectors = np.linalg.eigh(hamiltonian)
    shifted = energies - energies[0]
    if temperature == 0 or thermalisation:
        scale = max(1.0, float(np.max(np.abs(energies))))
        weights = (shifted <= 32 * np.finfo(float).eps * scale).astype(float)
    else:
        with np.errstate(over="ignore"):
            weights = np.exp(-shifted / temperature)
    weights /= np.sum(weights)
    return (vectors * weights) @ vectors.conj().T


def _gibbs_state_anharmonic(hamiltonian, temperature):
    """Gibbs state for H in ps^-1 and temperature in kelvin."""
    temperature = _real_scalar(
        temperature, "temperature", minimum=0.0
    )

    H = np.asarray(hamiltonian, dtype=complex)

    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] == 0:
        raise ValueError("hamiltonian must be a nonempty square matrix.")

    if not np.all(np.isfinite(H)) or not np.allclose(H, H.conj().T):
        raise ValueError("hamiltonian must be finite and Hermitian.")

    energies, vectors = np.linalg.eigh(H)

    if temperature == 0:
        # Select a ground state; the Morse bound-state ground level is unique.
        ground = vectors[:, 0]
        return np.outer(ground, ground.conj())

    hbar_meV_ps = 0.6582119512777485
    kb_meV_k = 0.08617333262145
    beta = hbar_meV_ps / (kb_meV_k * temperature)

    shifted = energies - energies[0]
    weights = np.exp(-beta * shifted)
    weights /= weights.sum()

    return (vectors * weights) @ vectors.conj().T


class AceMode:
    """A finite-dimensional environment mode for ACE.

    Parameters
    ----------
    hamiltonian : array_like
        Hermitian free-mode Hamiltonian in angular frequency units.
    coupling_operator : array_like
        Hermitian bath operator B in the interaction S (x) B, in angular
        frequency units. S is the separate system coupling operator.
    initial_state : array_like
        Positive semidefinite, Hermitian, trace-one mode density matrix.
        Different modes initially form a product state with the system.
    gammas : sequence of float, optional
        Nonnegative rates multiplying Lindblad dissipators D[L], one rate
        per entry in lindblad_operators.
    lindblad_operators : sequence of array_like, optional
        Dimensionless mode collapse operators L, which need not be Hermitian.
        The generator includes gamma * (L rho L.dagger -
        {L.dagger L, rho}/2). Operators may be complex.

    Notes
    -----
    Input arrays are copied and public array properties return copies.
    Temperature helper arguments mean k_B T / hbar, never Kelvin. Custom
    Hamiltonians can describe signed-frequency mapped modes, finite spins,
    anharmonic oscillators and arbitrary finite pseudomodes.
    """

    def __init__(self, hamiltonian, coupling_operator, initial_state,
                 gammas=None, lindblad_operators=None, thermalisation=False):
        hamiltonian = _matrix(hamiltonian, "hamiltonian", hermitian=True)
        self._dimension = hamiltonian.shape[0]
        coupling_operator = _matrix(
            coupling_operator, "coupling_operator", self.dimension, True)
        initial_state = _matrix(
            initial_state, "initial_state", self.dimension, True)
        if not np.isclose(np.trace(initial_state), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("initial_state must have trace one.")
        if np.min(np.linalg.eigvalsh(initial_state)) < -1e-12:
            raise ValueError("initial_state must be positive semidefinite.")
        if (gammas is None) != (lindblad_operators is None):
            raise ValueError("Supply gammas and lindblad_operators together.")
        if gammas is None:
            rates = np.empty(0, dtype=float)
            operators = ()
        else:
            try:
                rate_values = list(gammas)
                operator_values = list(lindblad_operators)
            except TypeError as exc:
                raise TypeError(
                    "gammas and lindblad_operators must be sequences.") from exc
            if len(rate_values) != len(operator_values):
                raise ValueError(
                    "gammas and lindblad_operators must have equal lengths.")
            rates = np.array([
                _real_scalar(rate, f"gammas[{index}]", minimum=0.0)
                for index, rate in enumerate(rate_values)], dtype=float)
            operators = tuple(
                _matrix(op, f"lindblad_operators[{index}]", self.dimension)
                for index, op in enumerate(operator_values))
        self._hamiltonian = hamiltonian
        self._coupling_operator = coupling_operator
        self._initial_state = initial_state
        self._gammas = rates
        self._lindblad_operators = operators
        self.thermalisation = bool(thermalisation)
        for array in (hamiltonian, coupling_operator, initial_state, rates,
                      *operators):
            array.setflags(write=False)

    @property
    def dimension(self):
        """Mode Hilbert space dimension."""
        return self._dimension

    @property
    def hamiltonian(self):
        """A copy of the free-mode Hamiltonian (angular frequency units)."""
        return self._hamiltonian.copy()

    @property
    def coupling_operator(self):
        """A copy of the bath coupling operator (angular frequency units)."""
        return self._coupling_operator.copy()

    @property
    def initial_state(self):
        """A copy of the initial density matrix."""
        return self._initial_state.copy()

    @property
    def gammas(self):
        """A copy of the Lindblad rates."""
        return self._gammas.copy()

    @property
    def lindblad_operators(self):
        """A tuple containing copies of the mode collapse operators."""
        return tuple(operator.copy() for operator in self._lindblad_operators)

    @classmethod
    def thermal(cls, hamiltonian, coupling_operator, temperature,
                gammas=None, lindblad_operators=None, thermalisation=False):
        """Gibbs state at temperature k_B T / hbar (not Kelvin).

        At zero temperature use the normalized ground-eigenspace projector.
        Subtract the ground energy before exponentiation to avoid overflow.

        If the bath is thermalised start the environment within it's ground
        state.

        """
        hamiltonian = _matrix(hamiltonian, "hamiltonian", hermitian=True)
        if thermalisation:
            # Number-basis vacuum, NOT the ground state of a negative H.
            state = np.zeros_like(hamiltonian)
            state[0, 0] = 1.0
        else:
            state = _gibbs_state(hamiltonian, temperature)
        return cls(hamiltonian, coupling_operator, state, gammas,
                   lindblad_operators, thermalisation=thermalisation)

    @classmethod
    def harmonic(cls, omega, coupling_strength, n_levels, temperature=0.0,
                 thermalisation=False):
        """Truncated harmonic mode H=omega*n, B=g*(a+a.dagger).
        omega must be real, g real, and n_levels a positive integer.
        Temperature is k_B T / hbar; zero produces the vacuum.
        Increase n_levels to check convergence of oscillator truncation.
        """
        omega = _real_scalar(omega, "omega")
        coupling_strength = _real_scalar(coupling_strength, "coupling_strength")
        n_levels = _levels(n_levels)
        number = np.arange(n_levels, dtype=float)
        annihilation = np.diag(np.sqrt(number[1:]), 1)
        return cls.thermal(
            np.diag(omega * number),
            coupling_strength * (annihilation + annihilation.T), temperature,
            thermalisation=thermalisation)

    @classmethod
    def anharmonic(cls, omega, coupling_strength, n_levels, anharmonicity,
                   temperature=0.0):
        """Number-anharmonic mode following the supplied ACE implementation.

        H=omega*(n+delta*n**2)/(1+delta), B=g*(a+a.dagger), with
        delta=anharmonicity. Nonnegative delta keeps the oscillator spectrum
        increasing as its basis is enlarged. For another finite spectrum use
        AceMode.thermal directly. Temperature is k_B T / hbar.
        """
        omega = _real_scalar(omega, "omega", minimum=0.0, strict=True)
        coupling_strength = _real_scalar(coupling_strength, "coupling_strength")
        delta = _real_scalar(anharmonicity, "anharmonicity", minimum=0.0)
        n_levels = _levels(n_levels)
        number = np.arange(n_levels, dtype=float)
        annihilation = np.diag(np.sqrt(number[1:]), 1)
        energies = omega * (number + delta * number**2) / (1 + delta)
        return cls.thermal(
            np.diag(energies),
            coupling_strength * (annihilation + annihilation.T), temperature)

    @classmethod
    def morse(cls, omega, coupling_strength, n_levels, depth=None,
              temperature=0.0, thermalisation=False):
        """Finite bound-state Morse mode with thermal mean-centered coupling.

        depth is the source model's dimensionless N parameter, defaulting to
        n_levels-1+0.1. It must exceed 1/2; each retained index must satisfy
        n < depth, so only bound states are included. Normalized energies are
        omega*[depth**2-(depth-n)**2]/(2*depth-1). Position elements follow the
        supplied Morse implementation, using digamma and logarithmic gamma
        functions. B=g*sqrt(2*depth)*(X-<X>*I), with <X> evaluated in the same
        truncated Gibbs state used for propagation. Temperature means
        k_B T / hbar.
        """
        omega = _real_scalar(omega, "omega", minimum=0.0, strict=True)
        coupling_strength = _real_scalar(coupling_strength, "coupling_strength")
        n_levels = _levels(n_levels)
        if depth is None:
            depth = n_levels - 1 + 0.1
        depth = _real_scalar(depth, "depth", minimum=0.5, strict=True)
        if n_levels - 1 >= depth:
            raise ValueError(
                "Morse bound states require n_levels - 1 < depth.")

        if thermalisation:
            raise ValueError('Turn off thermalisation for Morse environments')
        number = np.arange(n_levels, dtype=float)
        energies = omega * number * (2 * depth - number) / (2 * depth - 1)
        hamiltonian = np.diag(energies)
        state = _gibbs_state_anharmonic(hamiltonian, temperature)
        position = np.zeros((n_levels, n_levels), dtype=float)
        for n in range(n_levels):
            position[n, n] = (
                np.log(2 * depth + 1) + digamma(2 * depth - n + 1)
                - digamma(2 * depth - 2 * n + 1)
                - digamma(2 * depth - 2 * n))
            for m in range(n + 1, n_levels):
                log_ratio = (
                    np.log(depth - n) + np.log(depth - m)
                    + gammaln(2 * depth - m + 1) + gammaln(m + 1)
                    - gammaln(2 * depth - n + 1) - gammaln(n + 1))
                value = (2 * (-1)**(m - n + 1) * np.exp(log_ratio / 2)
                         / ((m - n) * (2 * depth - n - m)))
                position[n, m] = value
                position[m, n] = value
        mean = np.trace(position @ state).real
        coupling = coupling_strength * np.sqrt(2 * depth) * (
            position - mean * np.eye(n_levels))
        return cls(hamiltonian, coupling, state)

    @classmethod
    def harmonic_pseudomode(cls, omega, coupling_strength, n_levels, damping,
                            temperature=0.0, thermalisation=False, *,
                            damping_model="oscillator"):
        """One damped mode; coupling_strength is g, temperature is k_B*T/hbar.
        For the original dimensionless coupling lambda, supply
        g = omega * sqrt(lambda). Signed nonzero omega is supported at
        temperature=0 with an explicit vacuum initial state.
        """
        omega = _real_scalar(omega, "omega")
        temperature = _real_scalar(temperature, "temperature", minimum=0.0)
        damping = _real_scalar(damping, "damping", minimum=0.0)
        coupling_strength = _real_scalar(coupling_strength, "coupling_strength")
        n_levels = _levels(n_levels)
        if omega == 0:
            raise ValueError("omega must be nonzero.")
        if omega < 0 and temperature > 0:
            raise ValueError(
                "Negative-frequency pseudomodes require temperature=0.")
        if damping_model not in ("oscillator", "transitions"):
            raise ValueError(
                "damping_model must be 'oscillator' or 'transitions'.")
        number = np.arange(n_levels, dtype=float)
        annihilation = np.diag(np.sqrt(number[1:]), 1)
        hamiltonian = np.diag(omega * number)
        bath_operator = coupling_strength * (annihilation + annihilation.T)
        if temperature == 0 or thermalisation:
            state = np.zeros((n_levels, n_levels), dtype=complex)
            state[0, 0] = 1.0
        else:
            state = _gibbs_state(hamiltonian, temperature)
        if temperature == 0 or damping == 0:
            occupation = 0.0
        else:
            with np.errstate(over="ignore", divide="ignore"):
                occupation = 1.0 / np.expm1(omega / temperature)
        if not np.isfinite(occupation):
            raise ValueError(
                "Thermal occupation is not finite; check omega/temperature.")
        rates = []
        operators = []
        if damping_model == "oscillator":
            operators.extend((annihilation, annihilation.conj().T))
            rates.extend((damping * (occupation + 1), damping * occupation))
        else:
            for n in range(n_levels - 1):
                downward = np.zeros((n_levels, n_levels), dtype=complex)
                downward[n, n + 1] = np.sqrt(n + 1)
                operators.extend((downward, downward.conj().T))
                rates.extend((damping * (occupation + 1), damping * occupation))
        return cls(hamiltonian, bath_operator, state, rates, operators,
                   thermalisation=thermalisation)


def discrete_bath_correlation_function(omega, coupling_strengths, ti, tf, dt,
                                       thermalisation=False):

    """

    Function to calcualte the discretised bath correlation function
    from the paramaters for discretising the spectral density


    Paramaters
    ----------

    omega : ndarray
        Frequencies for the discretised spectral density
    coupling_strengths : ndarray
        Coupling strenghts for the discretised spectral density
    ti : float
        The initial time for propagation usually 0
    tf : float
        The final time for propagation
    dt : float
        The number of timesteps
    thermalisation : bool
        If false then need to account for temperature, or
        if true then standard fourier transform of the
        spectral density


    Returns
    -------

    bath_correlation_function
        The values for the bath correlation function for a given time
        window. This can be used to determine convergence with the bath
        correlation function for a given number of modes


    """


    ti = _real_scalar(ti, "ti", minimum=0.0)
    tf = _real_scalar(tf, 'tf', minimum=0.0)
    dt = _real_scalar(dt, 'dt', minimum=0.0)


    if tf <= ti:
        raise ValueError("tf must be greater than ti.")

    times = np.linspace(ti, tf, int(tf/dt))
    integral_real = np.sum(
        coupling_strengths[:, None]**2
        * np.cos(omega[:, None] * times[None, :]), axis=0)
    integral_imag = np.sum(
        coupling_strengths[:, None]**2
        * np.sin(omega[:, None] * times[None, :]), axis=0)


    return integral_real - 1j * integral_imag


def plot_bath_correlation_function(ti, tf, dt, bath_correlation_function):

    """

    Helper function to plot the bath correlation function


    Paramaters
    ----------

    ti : float
        The initial time for propagation usually 0
    tf : float
        The final time for propagation
    dt : float
        The number of timesteps
    bath_correlation_function : ndarray
        Function containin the bath correlation function values


    """

    ti = _real_scalar(ti, "ti", minimum=0.0)
    tf = _real_scalar(tf, 'tf', minimum=0.0)
    dt = _real_scalar(dt, 'dt', minimum=0.0)


    if tf <= ti:
        raise ValueError("tf must be greater than ti.")

    times = np.linspace(ti, tf, int(tf/dt))
    plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(times, np.real(bath_correlation_function))
    plt.plot(times, np.imag(bath_correlation_function))
    plt.xlabel('Time (ps)')
    plt.ylabel('C(t)')
    plt.show()

"""Spectral discretization with one public frequencies/couplings interface.

The callback is J(omega, temperature), including thermal factors. No automatic
thermalization, unit conversion, or extra pi factor is performed.
"""
from numbers import Real
import warnings

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln

__all__ = ["discretize_spectral_density"]


def _real_scalar(value, name, *, minimum=0.0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}.")
    return value


def _hard_cutoff(omega, cutoff_frequency):
    return (np.abs(omega) <= cutoff_frequency).astype(float)


def _exponential_cutoff(omega, cutoff_frequency):
    return np.exp(-np.abs(omega) / cutoff_frequency)


def _gaussian_cutoff(omega, cutoff_frequency):
    return np.exp(-(np.abs(omega) / cutoff_frequency)**2)


CUTOFF_DICT = {
    "hard": _hard_cutoff,
    "exponential": _exponential_cutoff,
    "gaussian": _gaussian_cutoff,
}


def _quadrature_samples(intervals, num_modes, density, *, midpoint):
    orders = ([num_modes] if len(intervals) == 1 else
              [num_modes // 2, num_modes - num_modes // 2])
    frequency_parts, weight_parts = [], []
    for (left, right), order in zip(intervals, orders):
        if midpoint:
            spacing = (right - left) / order
            frequencies = left + (np.arange(order) + 0.5) * spacing
            weights = np.full(order, spacing)
        else:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            half_width = (right - left) / 2
            frequencies = left + (nodes + 1) * half_width
            weights = weights * half_width
        frequency_parts.append(frequencies)
        weight_parts.append(weights)
    frequencies = np.concatenate(frequency_parts)
    weights = np.concatenate(weight_parts)
    couplings = np.sqrt(weights * density(frequencies))
    return frequencies, couplings


def _equidistant(intervals, num_modes, density, *, max_nodes, rtol):
    return _quadrature_samples(intervals, num_modes, density, midpoint=True)


def _gauss_legendre(intervals, num_modes, density, *, max_nodes, rtol):
    return _quadrature_samples(intervals, num_modes, density, midpoint=False)


def _chain_mapping(intervals, num_modes, density, *, max_nodes, rtol):
    # The supplied recurrence routines use interval labels starting at 1.
    # Every interval here is active, so the label does not select a density.
    def wf(x, i):
        return density(x)

    ab, _, _, success, _ = mcdis(
        num_modes, rtol, quadfinT, max_nodes, 2,
        len(intervals), intervals, wf, 0, 2,
    )
    if not success:
        raise RuntimeError(
            "Chain coefficients did not converge with "
            f"chain_max_nodes={max_nodes}."
        )
    frequencies, vectors = eigh_tridiagonal(
        ab[:, 0], np.sqrt(ab[1:, 1])
    )
    couplings = np.sqrt(ab[0, 1]) * np.abs(vectors[0, :])
    # These already include the spectral density; do not multiply by J again.
    return frequencies, couplings


DISCRETISATION_STRATEGY = {
    "equidistant": _equidistant,
    "gauss_legendre": _gauss_legendre,
    "chain_mapping": _chain_mapping,
}


def discretize_spectral_density(
    spectral_density,
    omega_min,
    omega_max,
    num_modes,
    *,
    temperature,
    thermalisation=False,
    discretisation_strategy="gauss_legendre",
    cutoff=None,
    cutoff_frequency=None,
    thermal_rtol=1e-16,
    thermal_factor=1.0,
    chain_max_nodes=5000,
    chain_rtol=1e7 * np.finfo(float).eps,
):
    """Return (frequencies, couplings), each of length num_modes.

    spectral_density(omega, temperature, thermalisation) must return a
    finite, nonnegative
    real scalar. It must already contain the desired thermal factors and
    support negative frequencies when temperature > 0. Temperature is passed
    unchanged to the callback; its units are defined by that callback.
    If thermal_rtol is enabled, temperature MUST be in kelvin and frequencies
    MUST be angular frequencies in ps^-1.

    At zero temperature integrate [omega_min, omega_max]. Otherwise integrate
    [-omega_max, -omega_min] and [omega_min, omega_max]. The bounds must obey
    0 <= omega_min < omega_max. Positive temperature requires num_modes >= 2.

    Strategies are 'equidistant' (midpoint rule), 'gauss_legendre', and
    'chain_mapping' (the supplied adaptive recurrence algorithm). The first
    two split the mode count between bands; chain mapping adapts the nodes
    to the combined measure and does not prescribe a count for each band.
    For disconnected bands, chain-derived nodes can lie in the central gap.

    An optional 'hard', 'exponential', or 'gaussian' cutoff multiplies J once.
    Its frequency scale defaults to omega_max. A hard cutoff also shortens
    the integration bands to avoid sampling beyond its known support.
    thermal_rtol optionally truncates the negative band using
    J(-u)/J(+u) = exp(-thermal_factor*hbar*u/(k_B*T)). It must be in (0, 1).
    thermal_factor must match the callback's convention; it only controls
    this truncation and does not modify the callback or thermalize its output.
    This is a tail-truncation heuristic, not a bound on chain coefficient error.
    An entirely excluded negative band is omitted. None disables this cutoff.
    No pi factor is inserted: sum(couplings**2 * f(frequencies)) approximates
    integral(J(omega, temperature) * cutoff(omega) * f(omega) d omega).

    chain_max_nodes and chain_rtol control only the chain strategy. Its
    convergence criterion is inherited from the supplied mcdis routine.
    """
    if not callable(spectral_density):
        raise TypeError("spectral_density must be callable.")
    omega_min = _real_scalar(omega_min, "omega_min")
    omega_max = _real_scalar(omega_max, "omega_max")
    temperature = _real_scalar(temperature, "temperature")
    num_modes = _positive_integer(num_modes, "num_modes")
    if omega_max <= omega_min:
        raise ValueError("omega_max must be greater than omega_min.")
    if temperature > 0 and num_modes < 2:
        raise ValueError(
            "At positive temperature, num_modes must be at least 2.")

    if discretisation_strategy is None:
        discretisation_strategy = "gauss_legendre"
    if (not isinstance(discretisation_strategy, str)
            or discretisation_strategy not in DISCRETISATION_STRATEGY):
        raise ValueError(
            f"Unknown discretisation strategy {discretisation_strategy!r}; "
            f"choose from {list(DISCRETISATION_STRATEGY)}."
        )
    if discretisation_strategy == "chain_mapping":
        chain_max_nodes = _positive_integer(chain_max_nodes, "chain_max_nodes")
        chain_rtol = _real_scalar(chain_rtol, "chain_rtol")
        if chain_rtol == 0:
            raise ValueError("chain_rtol must be positive.")

    cutoff_fn = None
    if cutoff is not None:
        if not isinstance(cutoff, str) or cutoff not in CUTOFF_DICT:
            raise ValueError(
                f"Unknown cutoff {cutoff!r}; choose from {list(CUTOFF_DICT)}.")
        cutoff_fn = CUTOFF_DICT[cutoff]
        cutoff_frequency = _real_scalar(
            omega_max if cutoff_frequency is None else cutoff_frequency,
            "cutoff_frequency",
        )
        if cutoff_frequency == 0:
            raise ValueError("cutoff_frequency must be positive.")
        if cutoff == "hard":
            omega_max = min(omega_max, cutoff_frequency)
            if omega_max <= omega_min:
                raise ValueError(
                    "The hard cutoff excludes the entire integration band.")
    elif cutoff_frequency is not None:
        raise ValueError("Specify cutoff when supplying cutoff_frequency.")

    if thermal_rtol is not None:
        thermal_rtol = _real_scalar(thermal_rtol, "thermal_rtol")
        if not 0 < thermal_rtol < 1:
            raise ValueError("thermal_rtol must be between 0 and 1.")
        thermal_factor = _real_scalar(thermal_factor, "thermal_factor")
        if thermal_factor == 0:
            raise ValueError("thermal_factor must be positive.")

    intervals = []
    if thermalisation:
        omega_negative_max = omega_max
        if thermal_rtol is not None:
            hbar_meV_ps = 0.6582119512777485
            kb_meV_k = 0.08617333262145
            beta = hbar_meV_ps / (kb_meV_k * temperature)
            omega_negative_max = min(
                omega_max, -np.log(thermal_rtol) / (thermal_factor * beta)
            )
        if omega_negative_max > omega_min:
            intervals.append((-omega_negative_max, -omega_min))
    intervals.append((omega_min, omega_max))
    intervals = np.array(intervals, dtype=float)

    def effective_density(frequencies):
        values = np.array([
            _real_scalar(
                spectral_density(float(w), temperature, thermalisation),
                "spectral_density value")
            for w in frequencies
        ])
        if cutoff_fn is not None:
            values *= cutoff_fn(frequencies, cutoff_frequency)
        return values

    sampler = DISCRETISATION_STRATEGY[discretisation_strategy]
    return sampler(intervals, num_modes, effective_density,
                   max_nodes=chain_max_nodes, rtol=chain_rtol)


# Numerical recurrence routines from the supplied implementation.

def _two_columns(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (number_of_rows, 2)")
    return result


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def gauss(N, ab):
    """Return N Gauss nodes and weights from monic recurrence coefficients."""
    N = _positive_integer(N, "N")
    ab = _two_columns(ab, "ab")
    if len(ab) < N:
        raise ValueError("input array ab too short")
    if not np.all(np.isfinite(ab[:N])) or np.any(ab[:N, 1] < 0):
        raise ValueError(
            "recurrence coefficients must be finite with nonnegative beta")
    if N == 1:
        return ab[:1].copy()
    # The Jacobi matrix is symmetric tridiagonal; no dense matrix is needed.
    nodes, vectors = eigh_tridiagonal(ab[:N, 0], np.sqrt(ab[1:N, 1]))
    return np.column_stack((nodes, ab[0, 1] * vectors[0, :] ** 2))


def _discrete_measure(N, xw):
    N = _positive_integer(N, "N")
    xw = _two_columns(xw, "xw")
    if not np.all(np.isfinite(xw)) or np.any(xw[:, 1] < 0):
        raise ValueError(
            "nodes and weights must be finite; weights must be nonnegative")
    xw = xw[xw[:, 1] != 0].copy()
    if N > len(xw):
        raise ValueError("N exceeds the number of positive-weight nodes")
    return xw


def lanczos(N, xw):
    """Discrete recurrence coefficients using the Gragg-Harrod RKPW routine.

    Adapted from the supplied Julia translation of W. B. Gragg and W. J.
    Harrod, Numerische Mathematik 44 (1984), 317-335.
    """
    xw = _discrete_measure(N, xw)
    Ncap = len(xw)
    p0 = xw[:, 0].copy()  # NumPy slices are views; Julia slices are copies.
    p1 = np.zeros(Ncap)
    p1[0] = xw[0, 1]

    for n in range(Ncap - 1):
        pn = xw[n + 1, 1]
        gam, sig, t = 1.0, 0.0, 0.0
        xlam = xw[n + 1, 0]
        for k in range(n + 2):
            rho = p1[k] + pn
            tmp = gam * rho
            tsig = sig
            if rho <= 0:
                gam, sig = 1.0, 0.0
            else:
                gam, sig = p1[k] / rho, pn / rho
            tk = sig * (p0[k] - xlam) - gam * t
            p0[k] -= tk - t
            t = tk
            pn = tsig * p1[k] if sig <= 0 else t**2 / sig
            p1[k] = tmp

    return np.column_stack((p0[:N], p1[:N]))


def stieltjes(N, xw):
    """Return the first N recurrence coefficients by discretized Stieltjes."""
    xw = _discrete_measure(N, xw)
    xw = xw[np.argsort(xw[:, 0], kind="stable")]
    x, w = xw.T
    s0 = np.sum(w)
    ab = np.zeros((N, 2))
    ab[0] = (np.dot(x, w) / s0, s0)
    p1, p2 = np.zeros(len(x)), np.ones(len(x))

    for k in range(N - 1):
        p0, p1 = p1, p2
        p2 = (x - ab[k, 0]) * p1 - ab[k, 1] * p0
        weighted_square = w * p2**2
        s1 = np.sum(weighted_square)
        s2 = np.dot(x, weighted_square)
        if s1 <= 0 or not np.isfinite(s1) or not np.isfinite(s2):
            raise FloatingPointError(
                f"Stieltjes breakdown at recurrence step {k + 1}")
        ab[k + 1] = (s2 / s1, s1 / s0)
        s0 = s1
    return ab


def r_jacobi(N, a=0.0, b=None):
    """Monic Jacobi coefficients for weight (1-x)^a (1+x)^b on [-1, 1].

    b defaults to a. Based on the supplied Laurie/Gautschi routine,
    translated to Julia by Thibaut Lacroix (2020).
    """
    N = _positive_integer(N, "N")
    a, b = float(a), float(a if b is None else b)
    if not np.isfinite(a + b) or a <= -1 or b <= -1:
        raise ValueError(
            "Jacobi parameters a and b must be finite and greater than -1")
    ab = np.empty((N, 2))
    ab[0] = ((b - a) / (a + b + 2),
             np.exp((a + b + 1) * np.log(2) + gammaln(a + 1)
                    + gammaln(b + 1) - gammaln(a + b + 2)))
    if N > 1:
        n = np.arange(1, N, dtype=float)
        nab = 2 * n + a + b
        ab[1:, 0] = (b**2 - a**2) / (nab * (nab + 2))
        ab[1, 1] = 4 * (a + 1) * (b + 1) / ((a + b + 2)**2 * (a + b + 3))
        n = np.arange(2, N, dtype=float)
        nab = 2 * n + a + b
        ab[2:, 1] = (4 * (n + a) * (n + b) * n * (n + a + b)
                     / (nab**2 * (nab + 1) * (nab - 1)))
    return ab


def trr(t, i, AB, wf):
    """Map quadrature to a finite interval (one-based i)."""
    left, right = AB[i - 1]
    x = ((right - left) * t[:, 0] + right + left) / 2
    return np.column_stack((x, (right - left) * t[:, 1] * wf(x, i) / 2))


def whole_line_transform(t, i, wf):
    """Julia's str: map quadrature to the entire real line."""
    u = t[:, 0]
    x = u / (1 - u**2)
    w = t[:, 1] * wf(x, i) * (1 + u**2) / (1 - u**2)**2
    return np.column_stack((x, w))


def rtr(t, i, AB, wf):
    """Map quadrature to [AB[i-1, 0], infinity)."""
    u = t[:, 0]
    x = AB[i - 1, 0] + (u + 1) / (1 - u)
    return np.column_stack((x, 2 * t[:, 1] * wf(x, i) / (1 - u)**2))


def ltr(t, i, AB, wf):
    """Map quadrature to (-infinity, AB[i-1, 1]]."""
    u = t[:, 0]
    x = -(1 - u) / (u + 1) + AB[i - 1, 1]
    return np.column_stack((x, 2 * t[:, 1] * wf(x, i) / (u + 1)**2))


def quadfinT(N, i, uv, mc, AB, wf):
    """Transform an N-point rule to component i and apply its weight."""
    AB = _two_columns(AB, "AB")
    if not 1 <= i <= mc or len(AB) != mc:
        raise ValueError("invalid interval index or interval count")
    if len(uv) != N:
        raise ValueError("N must equal the number of quadrature nodes")
    left, right = AB[i - 1]
    if np.isneginf(left) and np.isposinf(right):
        return whole_line_transform(uv, i, wf)
    if np.isneginf(left):
        return ltr(uv, i, AB, wf)
    if np.isposinf(right):
        return rtr(uv, i, AB, wf)
    return trr(uv, i, AB, wf)


def mcdis(N, eps0, quad, Nmax, idelta, mc, AB, wf, mp, irout, *, DM=None):
    """Refine a multicomponent discretization until beta coefficients converge.

    Returns (ab, Ncap, kount, success, uv). On failure Ncap and uv describe
    the last completed rule. DM explicitly supplies optional (node, weight)
    rows, replacing the undefined global DM in the original Julia code.
    The beta-only relative convergence criterion matches the source.
    """
    N = _positive_integer(N, "N")
    Nmax = _positive_integer(Nmax, "Nmax")
    mc = _positive_integer(mc, "mc")
    if (not np.isfinite(idelta) or idelta <= 0
            or not np.isfinite(eps0) or eps0 <= 0):
        raise ValueError("idelta and eps0 must be finite and positive")
    AB = _two_columns(AB, "AB")
    if len(AB) != mc or np.any(np.isnan(AB)) or np.any(AB[:, 0] >= AB[:, 1]):
        raise ValueError("AB must contain mc intervals with left < right")
    if not isinstance(mp, (int, np.integer)) or mp < 0:
        raise ValueError("mp must be a nonnegative integer")
    discrete = np.empty((0, 2)) if DM is None else _two_columns(DM, "DM")
    if len(discrete) != mp:
        raise ValueError("mp must match the number of rows in DM")

    ab = np.zeros((N, 2))
    Ncap = int(np.floor((2 * N - 1) / idelta))
    uv = np.empty((0, 2))
    kount = -1
    while True:
        kount += 1
        increment = 1 if kount <= 1 else 2**(kount // 5) * N
        next_cap = Ncap + increment
        if next_cap > Nmax:
            warnings.warn(f"Ncap exceeds Nmax in mcdis with irout = {irout}",
                          RuntimeWarning, stacklevel=2)
            return ab, len(uv), kount, False, uv
        Ncap = next_cap
        uv = gauss(Ncap, r_jacobi(Ncap, 0.0, 0.0))
        parts = [quad(Ncap, i, uv, mc, AB, wf) for i in range(1, mc + 1)]
        if mp:
            parts.append(discrete)
        xwm = np.vstack(parts)
        previous_beta = ab[:, 1].copy()
        ab = stieltjes(N, xwm) if irout == 1 else lanczos(N, xwm)
        if not np.all(np.isfinite(ab)) or np.any(ab[:, 1] <= 0):
            raise FloatingPointError(
                "invalid recurrence coefficients; check the measure and N")
        if np.all(np.abs(ab[:, 1] - previous_beta) <= eps0 * np.abs(ab[:, 1])):
            return ab, Ncap, kount, True, uv

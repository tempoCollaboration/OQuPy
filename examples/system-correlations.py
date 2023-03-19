#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

import oqupy
import oqupy.operators as op


# -----------------------------------------------------------------------------

omega_cutoff = 3.04
alpha = 0.126
temperature = 0.1309
initial_state=op.spin_dm("z-")

def gaussian_shape(t, area = 1.0, tau = 1.0, t_0 = 0.0):
    return area/(tau*np.sqrt(np.pi)) * np.exp(-(t-t_0)**2/(tau**2))

detuning = lambda t: 0.0 * t

t = np.linspace(-2,3,100)
Omega_t = gaussian_shape(t, area = np.pi/2.0, tau = 0.245)
Delta_t = detuning(t)

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(op.sigma("z")/2.0, correlations)


tempo_parameters = oqupy.TempoParameters(
    dt=0.10,
    epsrel=10**(-5),
    dkmax=20,
    add_correlation_time=2.0)

oqupy.helpers.plot_correlations_with_parameters(correlations, tempo_parameters)

start_time = -1.0
process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=start_time,
                                        end_time=1.0,
                                        parameters=tempo_parameters)

def hamiltonian_t(t, delta=0.0):
    return delta/2.0 * op.sigma("z") \
        + gaussian_shape(t, area = np.pi/2.0, tau = 0.245)/2.0 \
        * op.sigma("x")
system = oqupy.TimeDependentSystem(hamiltonian_t)


# -----------------------------------------------------------------------------

print("-------- Example A --------")

times_a, times_b, correlations = oqupy.compute_correlations(
        system=system,
        process_tensor=process_tensor,
        operator_a=op.sigma("x"),
        operator_b=op.sigma("z"),
        times_a=-0.5,
        times_b=0.5,
        time_order="full",
        initial_state=initial_state,
        start_time=start_time,
        progress_type="bar")

print(f"times_a = {times_a}")
print(f"times_b = {times_b}")
print("Correlation matrix:")
print(correlations)

control = oqupy.Control(2)
control.add_single(5, op.left_super(op.sigma("x")))

s_xy_list = []
t_list = []
dynamics = oqupy.compute_dynamics(
    system=system,
    process_tensor=process_tensor,
    control=control,
    start_time=start_time,
    initial_state=initial_state)
t, s_x = dynamics.expectations(op.sigma("x"))
_, s_y = dynamics.expectations(op.sigma("y"))
_, s_z = dynamics.expectations(op.sigma("z"))

plt.figure(2)

for i, s_xyz in enumerate([s_x, s_y, s_z]):
    plt.plot(t, s_xyz.real, color=f"C{i}", linestyle="solid")
    plt.plot(t, s_xyz.imag, color=f"C{i}", linestyle="dotted")
    plt.xlabel(r'$t/$ps')
    plt.ylabel(r'$<\sigma_{xyz}>$')
    plt.scatter(times_b[0],correlations[0, 0].real, color="C2", marker="o")
    plt.scatter(times_b[0],correlations[0, 0].imag, color="C2", marker="x")

# -----------------------------------------------------------------------------

print("-------- Example B --------")

times_a, times_b, correlations = oqupy.compute_correlations(
        system=system,
        process_tensor=process_tensor,
        operator_a=op.sigma("x"),
        operator_b=op.sigma("z"),
        times_a=(-0.4, 0.4),
        times_b=slice(8, 14),
        time_order="ordered",
        initial_state=initial_state,
        start_time=start_time,
        progress_type="bar")

print(f"times_a = {times_a}")
print(f"times_b = {times_b}")
print("Shape of correlations matrix:")
print(np.abs(correlations) >= 0.0)

plt.show()

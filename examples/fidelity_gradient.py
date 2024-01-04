#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0,'.')
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op


# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.2
num_steps = 20

# -- bath --
alpha = 0.08
omega_cutoff = 4.0
temperature = 1.6
pt_dkmax = 40
pt_epsrel = 1.0e-5

# -- initial and target state --
initial_state = op.spin_dm('down')
target_state = op.spin_dm('x-')

# -- initial parameter guess --
x0 = np.zeros(2*num_steps)
y0 = np.ones(2*num_steps) * (np.pi/2) / (dt*num_steps)
z0 = np.zeros(2*num_steps)


# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(
    alpha=alpha,
    zeta=1,
    cutoff=omega_cutoff,
    cutoff_type='exponential',
    temperature=temperature)
bath = oqupy.Bath(0.5 * op.sigma('y'), correlations)
pt_tempo_parameters = oqupy.TempoParameters(
    dt=dt,
    epsrel=pt_epsrel,
    dkmax=pt_dkmax)
process_tensor = oqupy.pt_tempo_compute(
    bath=bath,
    start_time=0.0,
    end_time=num_steps * dt,
    parameters=pt_tempo_parameters,
    progress_type='bar')

# --- Define parametrized system ----------------------------------------------

def hamiltonian(x, y, z):
    h = np.zeros((2,2),dtype='complex128')
    for var, var_name in zip([x,y,z], ['x', 'y', 'z']):
        h += var * op.sigma(var_name)
    return h

parametrized_system = oqupy.ParametrizedSystem(hamiltonian)

# --- Compute fidelity, dynamics, and fidelity gradient -----------------------

fidelity_dict = oqupy.fidelity_gradient(
        system=parametrized_system,
        initial_state=initial_state,
        target_state=target_state,
        process_tensor=process_tensor,
        parameters=(x0,y0,z0),
        return_fidelity=True,
        return_dynamics=True)

print(f"the fidelity is {fidelity_dict['fidelity']}")
print(f"the fidelity gradient is {fidelity_dict['gradient']}")
t, s_x = fidelity_dict['dynamics'].expectations(op.sigma("x"))
plt.plot(t,s_x)

# -----------------------------------------------------------------------------

embed()

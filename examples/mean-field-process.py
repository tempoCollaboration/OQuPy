#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

# ----------------------------------------------------------------------------

omega_cutoff = 5.0
a = 0.1
temperature = 0.1
initial_state = oqupy.operators.spin_dm("z-")
initial_field = 1.0 + 1.0j

# System parameters
gn = 0.2    # Twice Rabi splitting
gam_down = 0.01 # incoherent loss
gam_up   = 0.01 # incoherent gain
Sx=np.array([[0,0.5],[0.5,0]])
wc=0.0
kappa=0.1
end_time=1
def field_eom(t, states, field):
    sx_exp = np.matmul(Sx, states[0]).trace().real
    return -(1j*wc+kappa)*field - 1j*gn*sx_exp
def H_MF(t, field):
    return 2.0 * gn * np.real(field) * Sx

system = oqupy.TimeDependentSystemWithField(H_MF,
        #gammas = [gam_down, gam_up],
        #lindblad_operators = [oqupy.operators.sigma("-"), tempo.operators.sigma("+")]
        )

mean_field_system = oqupy.MeanFieldSystem([system], field_eom)

correlations = oqupy.PowerLawSD(alpha=a,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)

pt_tempo_parameters = oqupy.TempoParameters(
    dt=0.1,
    tcut=2.0,
    epsrel=10**(-7))

process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=end_time,
                                        parameters=pt_tempo_parameters,
                                        unique=True)
control = None 

mean_field_dynamics = oqupy.compute_dynamics_with_field(
    mean_field_system=mean_field_system,
    process_tensor_list=[process_tensor],
	initial_field=initial_field,
    control_list=[control],
    start_time=0.0,
    initial_state_list=[initial_state])
system_dynamics = mean_field_dynamics.system_dynamics[0]

print(f"The final time t = {mean_field_dynamics.times[-1]:.1f} " \
      + f"the field is {mean_field_dynamics.fields[-1]:.8g} and the state is:")
print(system_dynamics.states[-1])

t, s_x = system_dynamics.expectations(oqupy.operators.sigma("x")/2, real=True)
t, s_z = system_dynamics.expectations(oqupy.operators.sigma("z")/2, real=True)
t, fields = mean_field_dynamics.field_expectations()
fig, axes = plt.subplots(2, figsize=(9,10))
axes[0].plot(t, s_z)
axes[1].plot(t, np.abs(fields)**2)
axes[1].set_xlabel('t')
axes[0].set_ylabel('<Sz>')
axes[1].set_ylabel('n')
plt.tight_layout()

plt.show()

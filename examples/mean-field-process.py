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
def field_eom(t, state, field):
    sx_exp = np.matmul(Sx, state).trace().real
    return -(1j*wc+kappa)*field - 1j*gn*sx_exp
def H(t, field):
    #print(t, field)
    return 2.0 * gn * np.abs(field) * Sx

system = oqupy.TimeDependentSystemWithField(H,
        field_eom,
        #gammas = [gam_down, gam_up],
        #lindblad_operators = [oqupy.operators.sigma("-"), tempo.operators.sigma("+")]
        )
correlations = oqupy.PowerLawSD(alpha=a,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)

pt_tempo_parameters = oqupy.TempoParameters(
    dt=0.1,
    dkmax=20,
    epsrel=10**(-7))
process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=end_time,
                                        parameters=pt_tempo_parameters)
control = None #oqupy.Control(2)
dynamics = oqupy.compute_dynamics_with_field(
    system=system,
    process_tensor=process_tensor,
	initial_field=initial_field,
    control=control,
    start_time=0.0,
    initial_state=initial_state)
with np.printoptions(precision=4):
    for i,t in enumerate(dynamics._times):
        print('t = {:.1f}\nField {:.8g}\nState\n{}'.format(t, dynamics._fields[i], dynamics._states[i]))

t, s_x = dynamics.expectations(oqupy.operators.sigma("x")/2, real=True)
t, s_z = dynamics.expectations(oqupy.operators.sigma("z")/2, real=True)
t, fields = dynamics.field_expectations()
fig, axes = plt.subplots(2, figsize=(9,10))
axes[0].plot(t, s_z)
axes[1].plot(t, np.abs(fields)**2)
axes[1].set_xlabel('t')
axes[0].set_ylabel('<Sz>')
axes[1].set_ylabel('n')
plt.tight_layout()
#plt.savefig('out.png', bbox_inches='tight')
stacked_data = np.array([t, dynamics._fields, dynamics._states], dtype=object)
#np.save('oqupy.npy', stacked_data)

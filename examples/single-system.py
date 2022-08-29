#!/usr/bin/env python


import sys
sys.path.insert(0,'..')
import oqupy
import numpy as np
import matplotlib.pyplot as plt
from oqupy import contractions

alpha = 0.2
nuc = 0.15
T = 0.026
Omega = 0.3
omega0_1, omega0_2 = 0.0, 0.2
omegac = 0.0
kappa = 0.01
Gamma_down = 0.01
Gamma_up = 0.8 * Gamma_down

sigma_z = oqupy.operators.sigma("z")
sigma_plus = oqupy.operators.sigma("+")
sigma_minus = oqupy.operators.sigma("-")

def H_MF_1(t, a):
    return 0.5 * omega0_1 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)
def H_MF_2(t, a):
    return 0.5 * omega0_2 * sigma_z +\
        0.5 * Omega * (a * sigma_plus + np.conj(a) * sigma_minus)

fractions = [0.5]
def field_eom(t, state_list, field):
    return -(1j*omegac+kappa)*field + np.trace(state_list[0])


subsystem_1 = oqupy.TimeDependentSystemWithField(H_MF_1)
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=nuc,
                                cutoff_type='gaussian',
                                temperature=T)
bath = oqupy.Bath(0.5 * sigma_z, correlations)
initial_field = np.sqrt(0.05)
initial_state_1 = np.array([[0,0],[0,1]])
initial_state_list = [initial_state_1]

tempo_parameters = oqupy.TempoParameters(dt=0.2, dkmax=20, epsrel=10**(-4))
start_time = 0.0
end_time = 10

# Super-system object used in both methods
super_system = oqupy.MeanFieldSystem([subsystem_1], field_eom=field_eom)

# Using process tensor
#process_tensor = oqupy.pt_tempo_compute(bath=bath,
#                                        start_time=start_time,
#                                        end_time=end_time,
#                                        parameters=tempo_parameters)
#control_list = [oqupy.Control(subsystem_1.dimension)]
#super_system_dynamics_process = \
#        contractions.compute_dynamics_with_field(super_system,
#                initial_field=initial_field, 
#                initial_state_list=initial_state_list, 
#                start_time=start_time,
#                process_tensor_list = [process_tensor])
# Using tempo 

tempo_sys = oqupy.MeanFieldTempo(mean_field_system=super_system,
                        bath_list=[bath],
                        parameters=tempo_parameters,
                        initial_state_list=[initial_state_1],
                        initial_field=initial_field,
                        start_time=0.0)
super_system_dynamics_tempo = tempo_sys.compute(end_time=end_time)


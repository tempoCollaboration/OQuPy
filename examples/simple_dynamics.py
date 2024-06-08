#!/usr/bin/env python

import sys
sys.path.insert(0, '.')
import oqupy
import numpy as np
import matplotlib.pyplot as plt
sigma_x = oqupy.operators.sigma("x")
sigma_z = oqupy.operators.sigma("z")
up_density_matrix = oqupy.operators.spin_dm("z+")
Omega = 1.0
omega_cutoff = 5.0
alpha = 0.3

system = oqupy.System(0.5 * Omega * sigma_x)
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential')
bath = oqupy.Bath(0.5 * sigma_z, correlations)
tempo_parameters = oqupy.TempoParameters(dt=0.1, tcut=3.0, epsrel=10**(-4))

dynamics = oqupy.tempo_compute(system=system,
                               bath=bath,
                               initial_state=up_density_matrix,
                               start_time=0.0,
                               end_time=2.0,
                               parameters=tempo_parameters,
                               unique=True)
t, s_z = dynamics.expectations(0.5*sigma_z, real=True)
print(s_z)
plt.plot(t, s_z, label=r'$\alpha=0.3$')
plt.xlabel(r'$t\,\Omega$')
plt.ylabel(r'$\langle\sigma_z\rangle$')
#plt.savefig('simple_dynamics.png')
plt.show()

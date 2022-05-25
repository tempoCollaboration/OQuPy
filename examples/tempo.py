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
field = 1.0 + 1.0j

# System parameters
gn = 0.2    # Twice Rabi splitting
gam_down = 0.01 # incoherent loss
gam_up   = 0.01 # incoherent gain
Sx=np.array([[0,0.5],[0.5,0]])
wc=0.0
kappa=0.1
end_time=1#.1
def H(t):
    return 2.0 * gn * np.abs(field) * Sx
    
system = oqupy.TimeDependentSystem(H)
correlations = oqupy.PowerLawSD(alpha=a, 
                                zeta=1, 
                                cutoff=omega_cutoff, 
                                cutoff_type='gaussian', 
                                temperature=temperature)
bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)


tempo_parameters = oqupy.TempoParameters(dt=0.1, dkmax=20, epsrel=10**(-7))

tempo_sys = oqupy.Tempo(system=system,
                        bath=bath,
                        initial_state=initial_state,
                        start_time=0.0,
                        parameters=tempo_parameters)
dynamics = tempo_sys.compute(end_time=end_time)
with np.printoptions(precision=4):
    for i,t in enumerate(dynamics._times):
        print('t = {:.1f}\nState\n{}'.format(t,  dynamics._states[i]))


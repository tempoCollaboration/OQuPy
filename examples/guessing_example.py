#!/usr/bin/env python

import sys
sys.path.insert(0,'..')

import oqupy
import numpy as np
import matplotlib.pyplot as plt
sigma_x = oqupy.operators.sigma('x')
sigma_y = oqupy.operators.sigma('y')
sigma_z = oqupy.operators.sigma('z')
sigma_p = oqupy.operators.sigma('+')
sigma_m = oqupy.operators.sigma('-')
up_density_matrix = oqupy.operators.spin_dm('z+')
down_density_matrix = oqupy.operators.spin_dm("z-")
mixed_density_matrix = oqupy.operators.spin_dm("mixed")

t_endA = 3.0 #0.1 # 3.0
t_endB = 2.0 #-1.9 # 2.0
t_endC = 3.0 #1.0 # 3.0


# Example A - Spin Boson from quickstart tutorial
figA, axesA = plt.subplots(3,1,figsize=(4,6), constrained_layout=True)
omega_cutoff = 2.5 #  Halved
alpha = 0.3
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential')
bath = oqupy.Bath(0.5 * sigma_z, correlations)


# Omega = 1.0, gam_down = 0.0 are parameters in tutorial
Omegas = [1.0, 10.0, 10.0]
gam_downs = [0.0, 0.0, 5.0]

for i, Omega in enumerate(Omegas):
    axesA[i].set_ylabel(r'$\langle \sigma_z\rangle$')
    axesA[i].set_xlabel('$t$')
    system = oqupy.System(0.5 * Omega * sigma_x,
                          gammas=[gam_downs[i]],
                          lindblad_operators=[sigma_m])
    params_nosys = oqupy.guess_tempo_parameters(bath=bath,
                                                start_time=0.0,
                                                end_time=t_endA,
                                                tolerance=0.01)
    params_sys = oqupy.guess_tempo_parameters(system=system,
                                              bath=bath,
                                              start_time=0.0,
                                              end_time=t_endA,
                                              tolerance=0.01)
    params_ref = oqupy.TempoParameters(dt=0.005,
                                       tcut=None,
                                       epsrel=1e-6,
                                       name='reference parameters')
    print('dt (bath only) = {:.2g} dt (with sys) = {:.2g}'.format(
        params_nosys._dt, params_sys._dt))
    param_strs = ['nosys ($dt={:.2g}$)'.format(params_nosys._dt),
                  'sys ($dt={:.2g}$)'.format(params_sys._dt),
                  'ref ($dt={:.2g}$)'.format(params_ref._dt)]
    lines = ['-', '--', ':']
    all_params = [params_nosys, params_sys, params_ref]
    for params, param_str, ls in zip(all_params[:2], param_strs[:2], lines[:2]):
        dynamics = oqupy.tempo_compute(system=system,
                                       bath=bath,
                                       initial_state=up_density_matrix,
                                       start_time=0.0,
                                       end_time=t_endA,
                                       parameters=params)
        t, s_z = dynamics.expectations(0.5*sigma_z, real=True)
        axesA[i].plot(t, s_z, label=param_str, ls=ls)
    axesA[i].legend()

figA.savefig('exampleA.png', dpi=300, bbox_inches='tight')

# Example B - pulse from pt_tempo.ipynb

def gaussian_shape(t, area = 1.0, tau = 1.0, t_0 = 0.0):
    return area/(tau*np.sqrt(np.pi)) * np.exp(-(t-t_0)**2/(tau**2))

detuning = lambda t: 0.0 * t
omega_cutoff = 1.52 # halved
alpha = 0.126
temperature = 0.1309
initial_state=down_density_matrix

def hamiltonian_t(t):
    return detuning(t)/2.0 * sigma_z \
           + gaussian_shape(t, area = np.pi/2.0, tau = 0.1225)/2.0 * sigma_x # tau decreased from 0.245

# Aside - plot pulse
#figT, axT = plt.subplots()
#t = np.linspace(-2,3, num=400)
#t_sys = np.linspace(-2,3, num=196)
#t_nosys = np.linspace(-2,3, num=40)
#nosys_samples = gaussian_shape(t_nosys, area = np.pi/2.0, tau = 0.1)/2.0
#sys_samples = gaussian_shape(t_sys, area = np.pi/2.0, tau = 0.1)/2.0
#full_samples = gaussian_shape(t, area = np.pi/2.0, tau = 0.1)/2.0
#axT.plot(t_nosys, nosys_samples, label='{:.2g}'.format(t_nosys[1]-t_nosys[0]))
#axT.plot(t_sys, sys_samples, label='{:.2g}'.format(t_sys[1]-t_sys[0]))
#axT.plot(t, full_samples, label='{:.2g}'.format(t[1]-t[0]))
#axT.legend()
#axT.set_xlabel(r'$t$')
#figT.savefig('shape.png', bbox_inches='tight', dpi=300)

system = oqupy.TimeDependentSystem(hamiltonian_t)
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(sigma_z/2.0, correlations)
params_nosys = oqupy.guess_tempo_parameters(#
                                          bath=bath,
                                          start_time=-2.0,
                                          end_time=t_endB,
                                          tolerance=0.01)
params_sys = oqupy.guess_tempo_parameters(system=system,
                                          bath=bath,
                                          start_time=-2.0,
                                          end_time=t_endB,
                                          tolerance=0.01)
print('dt (bath only) = {:.2g} dt (with sys) = {:.2g}'.format(
        params_nosys._dt, params_sys._dt))

params_ref = oqupy.TempoParameters(dt=0.0125,
                                   tcut=None,
                                   epsrel=1e-6,
                                   name='reference parameters')
param_strs = ['nosys ($dt={:.2g}$)'.format(params_nosys._dt),
                  'sys ($dt={:.2g}$)'.format(params_sys._dt),
                  'ref ($dt={:.2g}$)'.format(params_ref._dt),
              ]
all_params = [params_nosys, params_sys, params_ref]
figB, axB = plt.subplots(constrained_layout=True)
axB.set_title(r'$\Delta=0.0\ \tau=0.1$')
axB.set_xlabel(r'$t$')
axB.set_ylabel(r'$\langle\sigma_{xy}\rangle$')
lines = ['-', '--', ':']
for params, param_str, ls in zip(all_params[:2], param_strs[:2],lines[:2]):
    tempo_sys = oqupy.Tempo(system=system,
                        bath=bath,
                        initial_state=initial_state,
                        start_time=-2.0,
                        parameters=params)
    dynamics = tempo_sys.compute(end_time=t_endB)
    t, s_x = dynamics.expectations(sigma_x, real=True)
    t, s_y = dynamics.expectations(sigma_y, real=True)
    s_xy = np.sqrt(s_x**2 + s_y**2)
    axB.plot(t, s_xy, label=param_str, ls=ls)
axB.set_ylim((0.0,1.0))
axB.legend(loc=4)
figB.savefig('exampleB.png', bbox_inches='tight', dpi=300)


# Example C - periodic t-dependent Ham

omega = 1.5
Delta = lambda t: 3.0 * np.sin(2*np.pi*omega*t)
def hC(t):
    return 3.0 * sigma_z \
           + Delta(t) * sigma_x 

alpha = 0.4
omega_cutoff = 0.4
system = oqupy.TimeDependentSystem(hC)
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=3,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(sigma_z/2.0, correlations)
params_nosys = oqupy.guess_tempo_parameters(#
                                          bath=bath,
                                          start_time=0.0,
                                          end_time=t_endC,
                                          tolerance=0.01)
params_sys = oqupy.guess_tempo_parameters(system=system,
                                          bath=bath,
                                          start_time=0.0,
                                          end_time=t_endC,
                                          tolerance=0.01)
params_ref = oqupy.TempoParameters(dt=0.005,
                                       tcut=None,
                                       epsrel=1e-6,
                                       name='reference parameters')
print('dt (bath only) = {:.2g} dt (with sys) = {:.2g}'.format(
        params_nosys._dt, params_sys._dt))

param_strs = ['nosys ($dt={:.2g}$)'.format(params_nosys._dt),
                  'sys ($dt={:.2g}$)'.format(params_sys._dt),
                  'ref ($dt={:.2g}$)'.format(params_ref._dt),
              ]
all_params = [params_nosys, params_sys, params_ref]
figC, axesC = plt.subplots(1,2,constrained_layout=True, figsize=(8,4))
axesC[0].set_xlabel(r'$t$')
axesC[1].set_xlabel(r'$t$')
ts = np.linspace(0, t_endC, num=300) 
axesC[0].plot(ts, Delta(ts))
axesC[0].set_ylabel(r'$\Delta(t)$')
axesC[1].set_ylabel(r'$\langle\sigma_{z}\rangle$')
lines = ['-', '--', ':']
for params, param_str, ls in zip(all_params, param_strs, lines):
    tempo_sys = oqupy.Tempo(system=system,
                        bath=bath,
                        initial_state=initial_state,
                        start_time=0.0,
                        parameters=params)
    dynamics = tempo_sys.compute(end_time=t_endC)
    t, s_z = dynamics.expectations(sigma_z, real=True)
    axesC[1].plot(t, s_z, label=param_str, ls=ls)
axesC[1].legend()
figC.savefig('exampleC.png', bbox_inches='tight', dpi=300)


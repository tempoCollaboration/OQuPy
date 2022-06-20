#!/usr/bin/env python

"""Script to calculate spectral weight and photoluminescence from two-time correlators"""

import pickle, sys

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', **{'size':14})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# input file
inputfp = 'data/dynamics_correlators.pkl'

# load
with open(inputfp, 'rb') as fb:
    data = pickle.load(fb)
times = data['times']
spsm = data['spsm']
smsp = data['smsp']
params = data['params']

dt = params['dt']
# calculations in rotating frame, plot in original frame
wc_rotating = params['wc'] + params['rotating_frame_freq']
w0_rotating = params['w0'] + params['rotating_frame_freq']
nus = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(times), d=dt))
plot_nus = 1e3 * (nus - params['rotating_frame_freq']) # units meV


# calculate self-energies
fft_smsp = np.fft.fftshift(dt * np.fft.ifft(smsp, norm='forward'))
fft_spsm_conjugate = np.fft.fftshift(dt * np.fft.ifft(np.conjugate(spsm), norm='forward'))
# Sigma^{-+}
energy_mp = -(1j/4)*params['gn']**2*(fft_smsp-fft_spsm_conjugate)
# Sigma^{--} 
energy_mm = -(1j/2)*params['gn']**2*(np.real(fft_smsp+fft_spsm_conjugate))


# calculate unperturbed Green's functions
# inverse of non-interacting retarded function
def D0RI(nu):
    return nu-wc_rotating+1j*params['kappa']
# N.B. DOIK = - DORI * DOK * DOA happens to be a constant
def D0IK(nu):
    return 2j * params['kappa']

# calculate interacting green's functions
inverse_retarded = D0RI(nus) - energy_mp
keldysh_inverse = D0IK(nus) - energy_mm
retarded = 1/inverse_retarded
advanced = np.conjugate(retarded)
keldysh = - retarded * keldysh_inverse * advanced

# calculate pl and spectral weight
#pl = (1j/2) * (keldysh - retarded + advanced)
# use explicit formula for PL in terms of self energies 
pl = (np.imag(energy_mp) - 0.5*np.imag(energy_mm)) \
        / np.abs((nus - wc_rotating) + 1j * params['kappa'] - energy_mp)**2
# spectral weight
spectral_weight = -2*np.imag(retarded)


# Plot correlators, spectral weight, photoluminescence 
fig1, axes1 = plt.subplots(2, figsize=(9,6), sharex=True)
fig2, ax2 = plt.subplots(figsize=(9,4))
fig3, ax3 = plt.subplots(figsize=(9,4))

axes1[1].set_xlabel('\(t\)')
axes1[0].plot(times, np.real(smsp), label=r'\(\text{Re}\langle \sigma^-(t) \sigma^+(0) \rangle\)')
axes1[0].plot(times, np.imag(smsp), label=r'\(\text{Im}\langle \sigma^-(t) \sigma^+(0) \rangle\)')
axes1[1].plot(times, np.real(spsm), label=r'\(\text{Re}\langle \sigma^+(t) \sigma^-(0) \rangle\)')
axes1[1].plot(times, np.imag(spsm), label=r'\(\text{Im}\langle \sigma^+(t) \sigma^-(0) \rangle\)')
axes1[0].legend()
axes1[1].legend()
ax2.set_xlabel(r'\(\nu\)')
ax2.set_ylabel(r'\(\varrho\)', rotation=0, labelpad=20)
ax2.plot(plot_nus, spectral_weight)
ax2.set_xlim([-300,300])
ax3.set_xlabel(r'\(\nu\)')
ax3.set_ylabel(r'\(\mathcal{L}\)', rotation=0, labelpad=20)
ax3.plot(plot_nus, pl)
ax3.set_xlim([-300,300])
fig1.savefig('figures/correlators.pdf', bbox_inches='tight')
fig2.savefig('figures/spectral_weight.pdf', bbox_inches='tight')
fig3.savefig('figures/photoluminescence.pdf', bbox_inches='tight')

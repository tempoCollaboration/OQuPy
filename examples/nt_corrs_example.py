#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:24:44 2024

@author: rmadw
"""

import sys
sys.path.insert(0,'..')

import oqupy
#from oqupy.contractions import compute_correlations_nt
import n_time_correlations as nt #with code implemented in oqupy, uncomment above and delete this import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title
from scipy.fft import fftfreq, fftshift, fft2, ifft
import os

os.chdir('/home/rmadw/Documents/OQuPy')
PT_DIR_PATH = "./tests/data/process_tensors/"


######################useful operators#########################################
P_1 = np.array([[0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.]], dtype=complex)

P_2 = np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 1.]], dtype=complex)

sigma_min = np.array([[0., 0., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]], dtype=complex)

sigma_plus = np.array([[0., 0., 0.],
                       [0., 0., 0.],
                       [0., 1., 0.]], dtype=complex)

######################compute the process tensor###############################
omega_cutoff = 3.04
alpha =0.1
temperature = 13.09 #=100 K
dt=0.1
start_time = 0.
end_time = dt*80
dkmax=500
epsrel=10**(-6)

tempo_parameters = oqupy.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)
syst_int = P_1  - P_2
correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(syst_int, correlations)

pt_file_path = os.path.join(PT_DIR_PATH, "3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel6.hdf5")
process_tensor = oqupy.import_process_tensor(pt_file_path)

######################define system + operators#########################
eps = 5.
omeg= 2.
reorg = 2.0*alpha*omega_cutoff
system = oqupy.System((eps+reorg)*(P_1 + P_2)
                      + omeg * (sigma_plus + sigma_min))

dip_v = np.array([[0., 0., 1.],
       [0., 0., 0.],[1., 0., 0.]], dtype=complex)

initial_state = np.array([[1., 0., 0.],
       [0., 0., 0.],[0., 0., 0.]], dtype=complex)

##########################Calculate four-time correlations###################
operators = [dip_v, dip_v, dip_v, dip_v]

order_1 = ["left", "right", "right", "left"]
order_2 = ["right", "left", "right", "left"]
order_3 =  ["right", "right", "left", "left"]
order_4 = ["left", "left", "left", "left"]

ops_orders = [order_1, order_2, order_3, order_4]

times_1 = (start_time, dt*40+dt)
times_2 = dt*40
times_3 = dt*40
times_4 = (dt*40, end_time)

ops_times = [times_1, times_2, times_3, times_4]

cors=[]

for i in range (len(ops_orders)):
    cor = nt.compute_nt_correlations(system = system,
                                      process_tensor=process_tensor,
                                      operators = operators,
                                      ops_times=ops_times,
                                      ops_order=ops_orders[i],
                                      dt = dt,
                                      initial_state = initial_state,
                                      start_time = start_time,
                                      progress_type = "bar")
    cors.append(cor)

##########################Calculate two-time correlations###################
order = ["left", "left"]

times = [start_time, (start_time, end_time)]

cor2 = nt.compute_nt_correlations(system = system,
                                  process_tensor=process_tensor,
                                  operators = [dip_v, dip_v],
                                  ops_times=times,
                                  ops_order=order,
                                  dt = dt,
                                  initial_state = initial_state,
                                  start_time = start_time,
                                  progress_type = "bar")

pad=500

t2 = cor2[0][1]

f=fftshift(ifft(cor2[1][0], n=(t2.size+pad)))

x=2*np.pi*fftshift(fftfreq(t2.size+pad,dt))


######################Plot#########################################
desired_folder = '/home/rmadw/Documents/OQuPy/examples'

Rs = []
for i in range (4):
    R = cors[i][1][:,0,0,:]
    R = R[::-1, :]
    Rs.append(R)

pad=500

Rfs=[]
for i in range (4):
    Rpad = np.pad(Rs[i], ((0,pad),(0,pad+1)), 'constant')
    Rf=fftshift((fft2(Rpad)))
    Rfs.append(Rf)

time = cors[0][0][0]
f_time = 2*np.pi*fftshift(fftfreq(time.size+pad,dt))

fig, ax = plt.subplots(1,1, figsize = (4/2.54, 3/2.54))
yax = np.flip(Rfs[0].real) + np.flip(Rfs[1].real,1) + np.flip(Rfs[2].real,1) + np.flip(Rfs[3].real)
cont1=ax.contour(f_time, f_time, yax, levels=15, linewidths=0.75)
cbar = fig.colorbar(cont1)
ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel(r'$\omega_{detec}\,\, (ps^{-1})$')
ax.set_ylabel(r'$\omega_{exc}\,\,(ps^{-1})$')
ax.plot([0, 1], [0, 1], '--', color='gray', transform=ax.transAxes, linewidth=0.75)
file_name = '2Dspectr_small_dt='+str(dt)+'_dkmax='+str(dkmax)+'_epsrel='+str(epsrel)+'_alph='+str(alpha)+'_temp='+str(np.round(temperature))+'omeg='+str(omeg)+'eps='+str(eps)+'wc='+str(omega_cutoff)+'t2='+str(0)+'.pdf'
full_path = os.path.join(desired_folder, file_name)
#plt.savefig(full_path, dpi=300, bbox_inches = "tight")

fig2, ax2 = plt.subplots(1,1, figsize = (4/2.54, 3/2.54))
ax2.plot(x, f, linewidth=1.5)
ax2.set_xlim([-10.,25.])
#ax2.set_ylim([-0.001,0.38])
ax2.set(xlabel=r'$\omega\,(ps^{-1})$', ylabel=r'Linear absorption (arb. units)')
file_name = '1Dspectr_small_dt='+str(dt)+'_dkmax='+str(dkmax)+'_epsrel='+str(epsrel)+'_alph='+str(alpha)+'_temp='+str(np.round(temperature))+'omeg='+str(omeg)+'eps='+str(eps)+'wc='+str(omega_cutoff)+'t2='+str(0)+'.pdf'
full_path = os.path.join(desired_folder, file_name)
#plt.savefig(full_path, dpi=300, bbox_inches = "tight")


figs, axs = plt.subplots(2,1, figsize = (5/2.54, 7/2.54))
axs[0].plot(x, f, linewidth=1.5)
axs[0].set(xlabel=r'$\omega\,(ps^{-1})$', ylabel=r'Linear absorption (arb. units)')
axs[0].set_xlim([-10, 25])
cont1=axs[1].contour(f_time, f_time, yax, levels=15, linewidths = 0.5)
axs[1].plot([0, 1], [0, 1], '--', color='gray', transform=axs[1].transAxes, linewidth=0.75)
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlim([-10, 25])
axs[1].set_ylim([-10, 25])
axs[1].set(xlabel = r'$\omega_{detec}$ (ps$^{-1})$', ylabel = r'$\omega_{exc}\,\,(ps^{-1})$')
cont1.monochrome = True
for col, ls in zip(cont1.collections, cont1._process_linestyles()):
    col.set_linestyle(ls)
norm= matplotlib.colors.Normalize(vmin=cont1.cvalues.min(), vmax=cont1.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = cont1.cmap)
sm.set_array([])
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sm, cax=cax, orientation='vertical')
file_name = 'spectr_dt='+str(dt)+'_dkmax='+str(dkmax)+'_epsrel='+str(epsrel)+'_alph='+str(alpha)+'_temp='+str(np.round(temperature))+'omeg='+str(omeg)+'eps='+str(eps)+'wc='+str(omega_cutoff)+'t2='+str(0)+'.pdf'
full_path = os.path.join(desired_folder, file_name)
#plt.savefig(full_path, dpi=300, bbox_inches = "tight")

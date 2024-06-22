# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to plot the multi-time correlations performance analysis results.
"""

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title
#plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle') # recommended

import dill

import oqupy
import oqupy.operators as op

import os

with open("./tests/data/performance_results/nt_corrs.pkl", 'rb') as f:
    all_results = dill.load(f)

# -----------------------------------------------------------------------------
pt4 = all_results[0][0][0]['corrs'][0,:]
pt6 = all_results[0][0][3]['corrs'][0,:]
pt8 = all_results[0][0][6]['corrs'][0,:]

r1 = (np.abs(pt8)-np.abs(pt4)).max()
r2 = (np.abs(pt8)-np.abs(pt6)).max()

fig, ax = plt.subplots(figsize=(8/2.54,6/2.54), tight_layout=True)
walltimes=[]
times=[]
for i in range(10):
    result = all_results[0][1][i]['walltime']
    N = all_results[0][1][i]['N']
    walltimes.append(result)
    times.append(N)
ax.plot(times, walltimes, '.')
ax.set_xlabel("Additional time steps")
ax.set_ylabel("Walltime (s)")
fig.savefig("./tests/data/plots/nt_corrs_time.pdf")

# -----------------------------------------------------------------------------

plt.show()

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
Skript to plot the bath dynamics performance analysis results.
"""

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import dill

import oqupy
import oqupy.operators as op

plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/bath_dynamics.pkl", 'rb') as f:
    all_results = dill.load(f)

styles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------

res = all_results[0][0][0]
walltime = res['walltime']
generate_correlations_time = res['generate_correlations_time']

epsilon = res['epsilon']
w_start, w_stop, w_step = res['w_start_stop_step']
ts = res['ts']
ws = res['ws']
energies = res['energies']
energiesEnvelope = res['energiesEnvelope']

w_index = int((epsilon - w_start)/w_step)

print(walltime)
print(generate_correlations_time)

# -----------------------------------------------------------------------------

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5.5,3.6), tight_layout=False, sharex=True)

ax1.plot(ts,energies[w_index]*1.0e3, color='k', linestyle=':')
ax1.plot(ts,energiesEnvelope[w_index]*1.0e3, color='k')
# ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$\Delta Q (\epsilon, t)\: [10^{-3}/\mathrm{ps}]$     ')

pcm = ax2.pcolormesh(ts, ws, energiesEnvelope*1.0e3, shading="nearest", cmap="coolwarm")
ax2.hlines(epsilon, ts[0], ts[-1], color='k', linestyle='-.')
ax2.set_xlabel(r'$t\quad[\mathrm{ps}]$')
ax2.set_ylabel(r'$\omega\quad[1/\mathrm{ps}]$')
fig.text(0.0,1.0,"(a)",fontsize=14,verticalalignment='top', horizontalalignment='left')
fig.text(0.0,0.53,"(b)",fontsize=14,verticalalignment='top', horizontalalignment='left')
fig.colorbar(pcm, label=r'$[10^{-3}/\mathrm{ps}]$', ax=ax2)

fig.savefig("./tests/data/plots/bath-dynamics-results.pdf")

# -----------------------------------------------------------------------------

plt.show()

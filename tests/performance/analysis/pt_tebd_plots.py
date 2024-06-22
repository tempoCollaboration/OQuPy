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
Skript to plot the PT-TEBD performance analysis results.
"""

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import dill

import oqupy
import oqupy.operators as op

plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/pt_tebd.pkl", 'rb') as f:
    all_results = dill.load(f)

styles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5.6,4))
for i in range(4):
    result = all_results[0][1][i]
    for site, dynamics in result['dynamics'].items():
        label = f"site $n$={site+1}" if i==0 else None
        ax1.plot(
            *dynamics.expectations(op.spin_dm("up"), real=True),
            color=f"C{site}", linestyle=styles[i],
            label=label)
ax1.legend(labelspacing=0.1, fontsize=10, handletextpad=0.3, borderpad=0.2)
ax1.set_xlim(0.0,4.0)
ax1.set_xlabel(r"$t\quad[\mathrm{ps}]$")
ax1.set_ylabel(r"$\langle\uparrow|\rho^{(n)}|\uparrow\rangle$")

for i in range(4):
    result = all_results[0][1][i]
    max_bond_dims = np.max(result["bond_dimensions"], 0)
    bonds = list(range(1,result['N']))
    walltime = result["walltime"]
    ax2.plot(bonds, max_bond_dims, color="#000000", linestyle=styles[i],
            label=f"K={result['K']} | walltime={result['walltime']:.01f}s")
ax2.legend(loc=8)
ax2.set_yscale('log')
ax2.set_ylim(1,200)
ax2.set_xticks(bonds)
ax2.set_xlabel("bond")
ax2.set_ylabel("bond dimension")
fig.savefig("./tests/data/plots/pt-tebd-results.pdf")

# -----------------------------------------------------------------------------

plt.show()

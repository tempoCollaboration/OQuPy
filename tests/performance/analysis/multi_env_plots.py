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
Skript to plot the multiple environments performance analysis results.
"""

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import dill

import oqupy
import oqupy.operators as op

plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/multi_env.pkl", 'rb') as f:
    all_results = dill.load(f)


# -----------------------------------------------------------------------------

results = all_results[0][0]


fig, ax = plt.subplots(1,1,figsize=(4,2.6))

colors = ['C0','C0','C3','C3','C2','C2']
linestyles = ['-','-',':',':','--','--']
labels = ['cold', None, 'hot', None, 'cold & hot', None]

for i, res in enumerate(results[0:6]):
    t, sz = res['dynamics'].expectations(op.sigma('z'), real=True)
    ax.plot(t, sz, color=colors[i], label=labels[i], linestyle=linestyles[i])

ax.set_xlim(left=t[0], right=t[-1])
ax.set_xlabel(r'$t\quad[\mathrm{ps}]$')
ax.set_ylabel(r'$\langle \sigma_z(t)\rangle$')
ax.legend()

fig.savefig("./tests/data/plots/multi-env-results.pdf")

# -----------------------------------------------------------------------------

plt.show()

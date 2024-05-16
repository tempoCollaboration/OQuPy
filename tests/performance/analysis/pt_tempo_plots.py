# Copyright 2024 The TEMPO Collaboration
#
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
from IPython import embed

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import dill

import oqupy
import oqupy.operators as op

plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/pt_tempo_E.pkl", 'rb') as f:
    all_results = dill.load(f)

# styles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------

alphas=[]
walltimePT=[]
walltimeDN=[] 
dynamics=[]
for res in all_results[0][0]:
    alphas.append(res['params']['alpha'])
    walltimePT.append(res['pt_tempo_walltime'])
    walltimeDN.append(res['dynamics_walltime'])
    dynamics.append(res['dynamics'])
alphas = np.array(alphas)
walltimePT = np.array(walltimePT)
walltimeDN = np.array(walltimeDN)

# -----------------------------------------------------------------------------

spin_dims=[]
walltimeA=[]
walltimeB=[] 
for res in all_results[1][0]:
    spin_dims.append(res['metadata']['parameters'][1])
    walltimeA.append(res['ptA_walltime'])
    walltimeB.append(res['ptB_walltime'])
spin_dims = np.array(spin_dims)
walltimeA = np.array(walltimeA)
walltimeB = np.array(walltimeB)

# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(5.6, 4.0))
axs = fig.subplot_mosaic([["deg", "pt"],
                          ["dyn", "dyn"]])

axs["deg"].semilogy((spin_dims-1)/2, walltimeA, label="without deg. checking",
                    c='k', marker="+", linestyle='none')
axs["deg"].semilogy((spin_dims-1)/2, walltimeB, label="with deg. checking",
                    c='k', marker="*", linestyle='none')
axs["deg"].set_xlabel("spin size")
axs["deg"].set_ylabel("computation time")
axs["deg"].set_ylim(top=1.0e3)
axs["deg"].legend(loc='upper center')


axs["pt"].semilogy(alphas, walltimePT, label="generation of PT-MPO",
                   c='k', marker="d", linestyle='none')
axs["pt"].semilogy(alphas, walltimeDN, label="application of PT-MPO",
                   c='k', marker="x", linestyle='none')
axs["pt"].set_xlabel(r"coupling strength $\alpha$")
axs["pt"].set_ylabel("computation time")
axs["pt"].set_ylim(bottom=1.0e-1, top=1.0e4)
axs["pt"].legend(loc='upper center')


for alpha, dyn in zip(alphas, dynamics):
    axs["dyn"].plot(*dyn.expectations(op.sigma('z'), real=True),label=f"$\\alpha = {alpha}$")
axs["dyn"].set_xlabel(r"$t\,/\mathrm{ps}$")
axs["dyn"].set_ylabel(r"$\langle\hat{\sigma}_z\rangle$")
axs["dyn"].set_ylim(-1.1,1.1)
axs["dyn"].legend()

# -----------------------------------------------------------------------------

fig.savefig("./tests/data/plots/pt-tempo-results.pdf")
plt.show()

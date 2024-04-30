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


fig, ax = plt.subplots(1,1)

for res in results[0:2]:
    ax.plot(*res['dynamics'].expectations(op.sigma('z'), real=True), color='C0', label="cold")

for res in results[2:4]:
    ax.plot(*res['dynamics'].expectations(op.sigma('z'), real=True), color='C3', label="warm")

for res in results[4:6]:
    ax.plot(*res['dynamics'].expectations(op.sigma('z'), real=True), color='C2', label="cold & warm")

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\langle \sigma_z(t)\rangle$')
ax.legend()

fig.savefig("./tests/data/plots/multi-env-results.pdf")

# -----------------------------------------------------------------------------

plt.show()

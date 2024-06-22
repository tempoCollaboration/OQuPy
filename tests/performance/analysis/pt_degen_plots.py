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
sys.path.insert(0, '.')

import oqupy.operators as op
import oqupy
from operator import itemgetter
import dill
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/pt_degen.pkl", 'rb') as f:
    all_results = dill.load(f)

styles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------

# PLOT
# -----------------------------------------------------------------------------

result_list = all_results[0]

unique_wall_times = []
unique_spin_array = []
non_unique_wall_times = []
non_unique_spin_array = []

for results in result_list:

    unique_wall_times.extend(list(map(itemgetter('walltime'),
                                      filter(itemgetter('unique'),
                                             results))))
    unique_spin_array.extend(list(map(itemgetter('spin_size'),
                                      filter(itemgetter('unique'),
                                             results))))
    non_unique_wall_times.extend(list(map(itemgetter('walltime'),
                                          filter(lambda x: not x.get('unique'),
                                                 results))))
    non_unique_spin_array.extend(list(map(itemgetter('spin_size'),
                                          filter(lambda x: not x.get('unique'),
                                                 results))))

fig, ax = plt.subplots()
ax.plot(unique_spin_array, unique_wall_times, label="on")
ax.plot(non_unique_spin_array, non_unique_wall_times, label="off")
ax.legend(title="Unique")
ax.set_yscale('log')
ax.set_ylabel("Walltime (s)")
ax.set_xlabel("Spin size")
fig.savefig("./tests/data/plots/pt-degen-results.pdf")

# -----------------------------------------------------------------------------


plt.show()

import matplotlib.pyplot as plt
import numpy as np
import dill
import sys
import os
sys.path.insert(0,'.')
plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

FIG_DIR_PATH = "./tests/data/plots/"

with open('./tests/data/performance_results/mean-field_results.pkl', 'rb') as f:
    all_results = dill.load(f)


test_a_results = all_results[0][0][0]

# plot runtime against number of systems
fig1, ax1 = plt.subplots(figsize=(4,2.0))
ax1.set_xlabel(r"Number of distinct systems $N_s$")
ax1.set_ylabel(r"Walltime (s)")
ax1.scatter(test_a_results['number_of_systems'][1:], test_a_results['walltimes'][1:])
ax1.scatter([0.5], [test_a_results['walltimes'][0]], c='red')
#ax1.scatter([1], [test_a_results['walltimes'][0]], c='red', marker="x")
ax1.set_ylim(bottom=0.0, top=None)
fig1.savefig(os.path.join(FIG_DIR_PATH, "mean-field-runtime-plot.pdf"))

# -----------------------------------------------------------------------------

plt.show()


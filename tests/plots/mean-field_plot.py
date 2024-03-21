import matplotlib.pyplot as plt
import numpy as np
import dill
import sys
import os
sys.path.insert(0,'.')
plt.style.use('./tests/plots/matplotlib_style.mplstyle')

FIG_DIR_PATH = "./tests/plots/figures/"

with open('./tests/data/temp/mean-field_results.pkl', 'rb') as f:
    all_results = dill.load(f)


test_a_results = all_results[0][0][0]
test_b_results = all_results[1][0][0]

# plot runtime against number of systems
number_of_systems_list, runtimes = test_a_results
fig1, ax1 = plt.subplots()
ax1.set_xlabel(r"Number of Systems")
ax1.set_ylabel(r"Runtime (s)")
ax1.scatter(number_of_systems_list, runtimes)
fig1.savefig(os.path.join(FIG_DIR_PATH, "mean-field-runtime-plot.png"))

# plot rescaled photon number against time
times, field_expectations = test_b_results
n = np.abs(field_expectations)**2 # get rescaled photon number

fig2, ax2 = plt.subplots()

ax2.set_xlabel(r'$t$ (ps)')
ax2.set_ylabel(r'$n/N$')
ax2.plot(times, n, label=r'$\Gamma_\uparrow = 0.8\Gamma_\downarrow$')
fig2.savefig(os.path.join(FIG_DIR_PATH, "mean-field-dynamics-plot.png"))


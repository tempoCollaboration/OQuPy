import matplotlib.pyplot as plt
import numpy as np
import dill

#include('matplotlib.style')

with open('./tests/data/temp/mean-field_results.pkl', 'rb') as f:
    all_results = dill.load(f)

# plot results
#fig/plt.savefig(./figs/)


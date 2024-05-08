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

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import dill

import oqupy
import oqupy.operators as op

plt.style.use('./tests/performance/analysis/matplotlib_style.mplstyle')

with open("./tests/data/performance_results/pt_degen.pkl", 'rb') as f:
    all_results = dill.load(f)

styles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------

# PLOT
# -----------------------------------------------------------------------------


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
Example of multi-time correlations.
"""
from IPython import embed

import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt

import oqupy
from oqupy import operators as op

# -----------------------------------------------------------------------------

sx = oqupy.operators.sigma("x")
sy = oqupy.operators.sigma("y")
sz = oqupy.operators.sigma("z")
up_dm = oqupy.operators.spin_dm("z+")
down_dm = oqupy.operators.spin_dm("z-")

# --- Parameters --------------------------------------------------------------

# -- time steps --
dt = 0.2
num_steps = 40

# -- bath --
alpha = 0.08
omega_cutoff = 4.0
temperature = 1.6
pt_dkmax = 40
pt_epsrel = 1.0e-5

# -- system --
epsilon = 1.0
Omega = 2.0
system = oqupy.System(0.5*epsilon*op.sigma('z')+ 0.5*Omega*op.sigma('x'))
initial_state = op.spin_dm('up')

# --- Compute process tensors -------------------------------------------------

correlations = oqupy.PowerLawSD(alpha=alpha,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature)
bath = oqupy.Bath(0.5 * sy, correlations)
pt_tempo_parameters = oqupy.TempoParameters(dt=dt,
                                            epsrel=pt_epsrel,
                                            dkmax=pt_dkmax)

print("Process tensor (PT) computation:")
pt = oqupy.pt_tempo_compute(bath=bath,
                            start_time=0.0,
                            end_time=num_steps * dt,
                            parameters=pt_tempo_parameters,
                            progress_type='bar')

# --- Compute two time correlations -------------------------------------------------

np.set_printoptions(precision=2)

print("---- A ----")
res = oqupy.compute_correlations(
    system,
    pt,
    operator_a = op.sigma('x'),
    operator_b = op.sigma('z'),
    times_a = slice(-5,None),
    times_b = slice(-5,None),
    time_order='ordered',
    initial_state=initial_state,
    progress_type='silent')
print(np.abs(res[-1]))

print("---- B ----")
res = oqupy.compute_correlations(
    system,
    pt,
    operator_a = op.sigma('x'),
    operator_b = op.sigma('z'),
    times_a = slice(0,5),
    times_b = slice(2,8),
    time_order='ordered',
    initial_state=initial_state,
    progress_type='silent')
print(np.abs(res[-1]))

print("---- C ----")
res = oqupy.compute_correlations(
    system,
    pt,
    operator_a = op.sigma('x'),
    operator_b = op.sigma('z'),
    times_a = slice(0,5),
    times_b = slice(2,8),
    time_order='anti',
    initial_state=initial_state)
print(np.abs(res[-1]))

# ----- Expected result ---------------

# Process tensor (PT) computation:
# --> PT-TEMPO computation:
# 100.0%   40 of   40 [########################################] 00:00:01
# Elapsed time: 1.5s
# ---- A ----
# [[0.1  0.2  0.29 0.37 0.45]
#  [ nan 0.11 0.2  0.27 0.35]
#  [ nan  nan 0.1  0.17 0.24]
#  [ nan  nan  nan 0.07 0.13]
#  [ nan  nan  nan  nan 0.03]]
# ---- B ----
# [[0.67 0.81 0.82 0.72 0.58 0.49]
#  [0.65 0.79 0.8  0.71 0.59 0.52]
#  [0.65 0.77 0.74 0.62 0.51 0.49]
#  [ nan 0.77 0.73 0.56 0.38 0.38]
#  [ nan  nan 0.72 0.56 0.3  0.21]]
# ---- C ----
# --> Compute correlations:
# 100.0%    2 of    2 [########################################] 00:00:00
# Elapsed time: 0.0s
# [[ nan  nan  nan  nan  nan  nan]
#  [ nan  nan  nan  nan  nan  nan]
#  [ nan  nan  nan  nan  nan  nan]
#  [0.59  nan  nan  nan  nan  nan]
#  [0.51 0.68  nan  nan  nan  nan]]
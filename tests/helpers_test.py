# Copyright 2021 The TEMPO Collaboration
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
Tests for the time_evovling_mpo.helpers module.
"""

import pytest
import numpy as np

import time_evolving_mpo as tempo


def test_plot_correlations_with_parameters():
    correlation_function = lambda t: (np.cos(t)+1j*np.sin(6.0*t)) * np.exp(-2.0*t)
    correlations = tempo.CustomCorrelations(correlation_function,
                                            max_correlation_time=10.0)
    param = tempo.TempoParameters(dt=0.1,
                                  dkmax=50,
                                  epsrel=3.9e-8)
    tempo.helpers.plot_correlations_with_parameters(correlations, param)

# Copyright 2020 The TimeEvolvingMPO Authors
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
Module to define global configuration for the time_evovling_mpo package.
"""

from numpy import float64, complex128

# Numpy datatype
NpDtype = complex128
NpDtypeReal = float64

# Seperator string for __str__ functions
SEPERATOR = "----------------------------------------------\n"

# The default backend for tensor network calculations
BACKEND = 'tensor-network'

# Dict of all backends and their default configuration
BACKEND_CONFIG = {
    'example': {'sleep_time':0.02},
    'tensor-network': {},
    }

# 'silent', 'simple' or 'bar' as a default to show the progress of computations
PROGRESS_TYPE = 'bar'

# relative precission for np.integrate.quad()
INTEGRATE_EPSREL = 2**(-26)

# maximal number of subdivision for adaptive np.integrate.quad()
SUBDIV_LIMIT = 256

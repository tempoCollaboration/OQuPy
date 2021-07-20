# Copyright 2021 The TimeEvolvingMPO Authors
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

# relative precission for np.integrate.quad()
INTEGRATE_EPSREL = 2**(-26)

# maximal number of subdivision for adaptive np.integrate.quad()
SUBDIV_LIMIT = 256

# 'silent', 'simple' or 'bar' as a default to show the progress of computations
PROGRESS_TYPE = 'bar'


# -- TEMPO --------------------------------------------------------------------

# The default tempo backend for tensor network calculations
TEMPO_BACKEND = 'tensor-network'

# Dict of all tempo backends and their default configuration
TEMPO_BACKEND_CONFIG = {
    'tensor-network': {'backend':'numpy'},
    }

# maximal dkmax for tempo parameter guessing function
MAX_DKMAX = 256

# default tolerance for tempo parameter guessing function
DEFAULT_TOLLERANCE = 3.9e-3


# -- PT_TEMPO -----------------------------------------------------------------

# The default process tensor tempo backend for tensor network calculations
PT_TEMPO_BACKEND = 'tensor-network'

# Dict of all process tensor tempo backends and their default configuration
PT_TEMPO_BACKEND_CONFIG = {
    'tensor-network': {},
    }

# maximal dkmax for process tensor tempo parameter guessing function
PT_MAX_DKMAX = 256

# default tolerance for process tensor tempo parameter guessing function
PT_DEFAULT_TOLLERANCE = 3.9e-3

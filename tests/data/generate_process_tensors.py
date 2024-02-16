# Copyright 2020 The TEMPO Collaboration
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
Skript to generate process tensors for testing purposes.
"""

import numpy as np
import oqupy

"""
ToDo: Generate process tensors that will then be made available for testing
      purposes under /tests/data/process_tensors.
      The generated process tensors should be stored on a public data record,
      (e.g. Zenodo) such that they can be loaded into
      /test/data/process_tensors when needed.

      The process tensors to generate should be combinations of
      Coupling strength alpha: 0.08 / 0.16 / 0.32
      Spectral density exponent: 0.5 / 1.0 / 3.0
      Temperature: 0.0 / 0.8 / 1.6 / 3.2
      Cutoff: gauss
      dt: 0.02 / 0.04 / 0.08

      The filenames should then have a form like this example:
      "boson_alpha0.16_zeta3.0_T0.8_gauss_dt0.04.hdf5"
"""


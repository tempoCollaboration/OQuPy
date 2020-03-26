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
Tests for the time_evovling_mpo.operators module.
"""

import pytest
import numpy as np

from time_evolving_mpo import operators

PAULI = {"id":[[1, 0], [0, 1]],
         "x":[[0, 1], [1, 0]],
         "y":[[0, -1j], [1j, 0]],
         "z":[[1, 0], [0, -1]]}

SPIN_DM = {"up":[[1, 0], [0, 0]],
           "down":[[0, 0], [0, 1]],
           "plus":[[0.5, 0.5], [0.5, 0.5]],
           "minus":[[0.5, -0.5], [-0.5, 0.5]],
           "plus-y":[[0.5, -0.5j], [0.5j, 0.5]],
           "minus-y":[[0.5, 0.5j], [-0.5j, 0.5]],
           "mixed":[[0.5, 0.0], [0.0, 0.5]]}

def test_identity():
    for n in {1,2,7}:
        result = operators.identity(n)
        np.testing.assert_almost_equal(result,np.identity(n))

def test_operators_pauli():
    for name in PAULI:
        result = operators.pauli(name)
        np.testing.assert_equal(result,PAULI[name])

def test_operators_spin_dm():
    for name in SPIN_DM:
        result = operators.spin_dm(name)
        np.testing.assert_equal(result,SPIN_DM[name])

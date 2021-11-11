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
Tests for the time_evovling_mpo.operators module.
"""

import pytest
import numpy as np

from oqupy import operators

SIGMA = {"id":[[1, 0], [0, 1]],
         "x":[[0, 1], [1, 0]],
         "y":[[0, -1j], [1j, 0]],
         "z":[[1, 0], [0, -1]],
         "+":[[0, 1], [0, 0]],
         "-":[[0, 0], [1, 0]]}

SPIN_DM = {"up":[[1, 0], [0, 0]],
           "down":[[0, 0], [0, 1]],
           "z+":[[1, 0], [0, 0]],
           "z-":[[0, 0], [0, 1]],
           "x+":[[0.5, 0.5], [0.5, 0.5]],
           "x-":[[0.5, -0.5], [-0.5, 0.5]],
           "y+":[[0.5, -0.5j], [0.5j, 0.5]],
           "y-":[[0.5, 0.5j], [-0.5j, 0.5]],
           "mixed":[[0.5, 0.0], [0.0, 0.5]]}


def test_identity():
    for n in {1,2,7}:
        result = operators.identity(n)
        np.testing.assert_almost_equal(result,np.identity(n))

def test_operators_sigma():
    for name in SIGMA:
        result = operators.sigma(name)
        np.testing.assert_equal(result,SIGMA[name])

def test_operators_spin_dm():
    for name in SPIN_DM:
        result = operators.spin_dm(name)
        np.testing.assert_equal(result,SPIN_DM[name])

def test_operators_create_destroy():
    for n in {1,2,7}:
        adag = operators.create(n)
        a = operators.destroy(n)
        result = adag@a
        np.testing.assert_almost_equal(result,np.diag(range(n)))

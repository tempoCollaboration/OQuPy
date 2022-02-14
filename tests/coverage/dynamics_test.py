# Copyright 2022 The TEMPO Collaboration
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
Tests for the time_evovling_mpo.dynamics module.
"""

import pytest
import numpy as np

from oqupy.dynamics import Dynamics


times = [0.0, 0.1]
states = [np.array([[1, 0], [0, 0]]), np.array([[0.9, -0.1j], [0.1j, 0.1]])]
other_time = 0.05
other_state = np.array([[0.99, -0.01j], [0.01j, 0.01]])


def test_dynamics():
    dyn_A = Dynamics()
    str(dyn_A)
    assert len(dyn_A) == 0
    dyn_B = Dynamics(times, states)
    str(dyn_B)
    assert len(dyn_B) == len(times)
    np.testing.assert_almost_equal(times, dyn_B.times)
    np.testing.assert_almost_equal(states, dyn_B.states)
    np.testing.assert_almost_equal(states[0].shape, dyn_B.shape)
    dyn_B.add(other_time, other_state)

def test_dynamics_bad_input():
    with pytest.raises(AssertionError):
        Dynamics(0.1, states)
    with pytest.raises(AssertionError):
        Dynamics(times, 0.1)
    with pytest.raises(AssertionError):
        Dynamics([other_time], states)
    dyn_A = Dynamics(times, states)
    with pytest.raises(AssertionError):
        dyn_A.add("bla", other_state)
    with pytest.raises(AssertionError):
        dyn_A.add(other_time, "bla")
    with pytest.raises(AssertionError):
        dyn_A.add(other_time, np.random.rand(3,3))
    dyn_B = Dynamics()
    with pytest.raises(AssertionError):
        dyn_A.add(other_time, np.random.rand(2,2,2))
    with pytest.raises(AssertionError):
        dyn_A.add(other_time, np.random.rand(3,2))

def test_dynamics_expectations():
    dyn_A = Dynamics()
    t, x = dyn_A.expectations()
    assert t is None
    assert x is None
    dyn = Dynamics(times, states)
    t, tr = dyn.expectations()
    np.testing.assert_almost_equal(t, dyn.times)
    np.testing.assert_almost_equal(tr, [1.0, 1.0])
    t, y = dyn.expectations([[0,-1j],[1j,0]],real=True)
    np.testing.assert_almost_equal(t, dyn.times)
    np.testing.assert_almost_equal(y, [0, 0.2])
    t, lower = dyn.expectations([[0,0],[1,0]])
    np.testing.assert_almost_equal(t, dyn.times)
    np.testing.assert_almost_equal(lower, [0, -0.1j])
    with pytest.raises(AssertionError):
        dyn.expectations("bla")

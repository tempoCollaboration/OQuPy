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
Tests for the time_evovling_mpo.system module.
"""

import pytest
import numpy as np

from oqupy.system import System
from oqupy.system import TimeDependentSystem
from oqupy import operators
from oqupy.system import BaseSystem

# -----------------------------------------------------------------------------
# -- test-examples ------------------------------------------------------------

# -- example A --
hamiltonianA = 0.4*operators.sigma("x")
liouvillianA = np.array([[0.-0.j , 0.+0.4j, 0.-0.4j, 0.-0.j ],
                         [0.+0.4j, 0.-0.j , 0.-0.j , 0.-0.4j],
                         [0.-0.4j, 0.-0.j , 0.-0.j , 0.+0.4j],
                         [0.-0.j , 0.-0.4j, 0.+0.4j, 0.-0.j ]])

# -- example B --
hamiltonianB = operators.sigma("z")
gammasB = [0.2, 0.1, 0.05]
lindblad_operatorsB = [operators.sigma("x"),
                       operators.sigma("y"),
                       operators.sigma("z")]
liouvillianB = np.array([[-0.3+0.j,  0. +0.j,  0. +0.j,  0.3+0.j],
                         [ 0. +0.j, -0.4-2.j,  0.1+0.j,  0. +0.j],
                         [ 0. +0.j,  0.1+0.j, -0.4+2.j,  0. +0.j],
                         [ 0.3+0.j,  0. +0.j,  0. +0.j, -0.3+0.j]])

# -- example C --
hamiltonianC = lambda t: t*operators.sigma("x")
timeC = 2.0
liouvillianC = np.array([[0.-0.j, 0.+2.j, 0.-2.j, 0.-0.j],
                         [0.+2.j, 0.-0.j, 0.-0.j, 0.-2.j],
                         [0.-2.j, 0.-0.j, 0.-0.j, 0.+2.j],
                         [0.-0.j, 0.-2.j, 0.+2.j, 0.-0.j]])

# -- example D --
hamiltonianD = lambda t: t*operators.sigma("z")
timeD = 2.0
gammasD = [lambda t: t*0.2, lambda t: t*0.1, lambda t: t*0.05]
lindblad_operatorsD = [lambda t: t*operators.sigma("x"),
                       lambda t: t*operators.sigma("y"),
                       lambda t: t*operators.sigma("z")]
liouvillianD = np.array([[-2.4+0.j,  0. +0.j,  0. +0.j,  2.4+0.j],
                         [ 0. +0.j, -3.2-4.j,  0.8+0.j,  0. +0.j],
                         [ 0. +0.j,  0.8+0.j, -3.2+4.j,  0. +0.j],
                         [ 2.4+0.j,  0. +0.j,  0. +0.j, -2.4+0.j]])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def test_base_system():
    dimension = 3
    sys = BaseSystem(dimension)
    assert sys.dimension == 3

    with pytest.raises(NotImplementedError):
        sys.liouvillian()

def test_system_A():
    sys = System(hamiltonianA)
    liouvillian = sys.liouvillian()
    np.testing.assert_almost_equal(liouvillian, liouvillianA)

def test_system_B():
    sys = System(hamiltonianB,
                 gammasB,
                 lindblad_operatorsB,
                 name="bla",
                 description="blub")
    str(sys)
    assert isinstance(sys.hamiltonian, np.ndarray)
    assert isinstance(sys.gammas, list)
    assert sys.gammas[0] == gammasB[0]
    assert isinstance(sys.lindblad_operators, list)
    assert isinstance(sys.lindblad_operators[0], np.ndarray)
    assert sys.dimension == 2
    liouvillian = sys.liouvillian()
    np.testing.assert_almost_equal(liouvillian, liouvillianB)

def test_system_bad_input():
    with pytest.raises(AssertionError):
        System("bla")
    with pytest.raises(AssertionError):
        System(np.random.rand(2,2,2))
    with pytest.raises(AssertionError):
        System(np.random.rand(2,3))
    with pytest.raises(AssertionError):
        System(operators.sigma("z"),
               0.1,
               [operators.sigma("x"),
                operators.sigma("y"),
                operators.sigma("z")])
    with pytest.raises(AssertionError):
        System(operators.sigma("z"),
               ["bla",0.1,0.05],
               [operators.sigma("x"),
                operators.sigma("y"),
                operators.sigma("z")])
    with pytest.raises(AssertionError):
        System(operators.sigma("z"),
               [0.2,0.1,0.05],
               0.1)
    with pytest.raises(AssertionError):
        System(operators.sigma("z"),
               [0.2,0.1,0.05],
               [operators.sigma("x"),
                "bla",
                operators.sigma("z")])

def test_time_dependent_system_C():
    sys = TimeDependentSystem(hamiltonianC)
    liouvillian = sys.liouvillian(timeC)
    np.testing.assert_almost_equal(liouvillian, liouvillianC)

def test_time_dependent_system_D():
    sys = TimeDependentSystem(
            hamiltonianD,
            gammasD,
            lindblad_operatorsD,
            name="bla",
            description="blub")
    with pytest.raises(ValueError):
        sys.liouvillian()

    str(sys)
    assert isinstance(sys.hamiltonian, np.vectorize)
    assert isinstance(sys.gammas, list)
    assert sys.gammas[0](1.0) == gammasD[0](1.0)
    assert isinstance(sys.lindblad_operators, list)
    assert isinstance(sys.lindblad_operators[0], np.vectorize)
    assert sys.dimension == 2
    liouvillian = sys.liouvillian(timeD)
    np.testing.assert_almost_equal(liouvillian, liouvillianD)


def test_time_dependent_system_bad_input():
    with pytest.raises(AssertionError):
        TimeDependentSystem("bla")
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*np.random.rand(2,2,2))
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*np.random.rand(2,3))
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*operators.sigma("z"),
               0.1,
               [lambda t: t*operators.sigma("x"),
                lambda t: t*operators.sigma("y"),
                lambda t: t*operators.sigma("z")])
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*operators.sigma("z"),
               [lambda t: t*0.2, 0.1, lambda t: t*0.05],
               [lambda t: t*operators.sigma("x"),
                lambda t: t*operators.sigma("y"),
                lambda t: t*operators.sigma("z")])
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*operators.sigma("z"),
               [lambda t: t*0.2, lambda t: t*0.1, lambda t: t*0.05],
               0.1)
    with pytest.raises(AssertionError):
        TimeDependentSystem(lambda t: t*operators.sigma("z"),
               [lambda t: t*0.2, lambda t: t*0.1, lambda t: t*0.05],
               [lambda t: t*operators.sigma("x"),
                operators.sigma("y"),
                lambda t: t*operators.sigma("z")])

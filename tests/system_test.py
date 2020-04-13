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
Tests for the time_evovling_mpo.system module.
"""

import pytest
import numpy as np

from time_evolving_mpo.system import System
from time_evolving_mpo.system import TimeDependentSystem
from time_evolving_mpo import operators
from time_evolving_mpo.system import BaseSystem


def test_base_system():
    dimension = 3
    sys = BaseSystem(dimension)
    sys.dimension

    with pytest.raises(NotImplementedError):
        sys.liouvillian()

def test_system():
    sys_A = System(0.4*operators.sigma("x"))
    sys_A.liouvillian()

    sys_B = System(operators.sigma("z"),
                   [0.2,0.1,0.05],
                   [operators.sigma("x"),
                    operators.sigma("y"),
                    operators.sigma("z")],
                   name="bla",
                   description="blub",
                   description_dict={"the answer":42})
    sys_B.liouvillian()
    str(sys_B)
    assert isinstance(sys_B.hamiltonian, np.ndarray)
    assert isinstance(sys_B.gammas, list)
    assert sys_B.gammas[0] == 0.2
    assert isinstance(sys_B.lindblad_operators, list)
    assert isinstance(sys_B.lindblad_operators[0], np.ndarray)

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

def test_time_dependent_system():
    sys_A = TimeDependentSystem(lambda t: t*operators.sigma("x"))
    sys_A.liouvillian(2.0)

    sys_B = TimeDependentSystem(lambda t: t*operators.sigma("z"),
                   [lambda t: t*0.2, lambda t: t*0.1, lambda t: t*0.05],
                   [lambda t: t*operators.sigma("x"),
                    lambda t: t*operators.sigma("y"),
                    lambda t: t*operators.sigma("z")],
                   name="bla",
                   description="blub",
                   description_dict={"the answer":42})
    with pytest.raises(ValueError):
        sys_B.liouvillian()
    sys_B.liouvillian(2.0)
    str(sys_B)
    assert isinstance(sys_B.hamiltonian, np.vectorize)
    assert isinstance(sys_B.gammas, list)
    assert sys_B.gammas[0](1.0) == 0.2
    assert isinstance(sys_B.lindblad_operators, list)
    assert isinstance(sys_B.lindblad_operators[0], np.vectorize)

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

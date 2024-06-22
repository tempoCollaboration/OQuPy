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

from oqupy.system import BaseSystem, System, TimeDependentSystem,\
        TimeDependentSystemWithField, MeanFieldSystem, ParameterizedSystem
from oqupy import operators
from typing import Callable

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

# -- field example --
hamiltonianFieldGood = lambda t, field: t*operators.sigma("z") + field
hamiltonianFieldBad = lambda t: t*operators.sigma("z")
hamiltonianFieldBad2 = lambda t, state: np.flatten(operators.sigma("z"))
fieldEomGood = lambda t, states, field: t*field + states[0].trace()
fieldEomBad = lambda t, states: 5*t
fieldEomBad2 = lambda t, states, field: t*states
timeField = 2.0
dtField = 0.2
fieldField = 1.0j
derivativeField = 1+2j
stateField = np.array([[0.5,0.1j],[-0.1j,0.5]])
liouvillianField = np.array(
[[ 0. -0.j,  -1.4+0.2j,  1.4-0.2j,	0. -0.j ],
[-1.4+0.2j,  0. -4.4j,	0. -0.j ,  1.4-0.2j],
[ 1.4-0.2j,  0. -0.j ,	0. +4.4j, -1.4+0.2j],
[ 0. -0.j ,  1.4-0.2j, -1.4+0.2j,  0. -0.j ]])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def test_base_system():
    dimension = 3
    name = 'Test base'
    sys = BaseSystem(dimension, name)
    assert sys.dimension == 3
    str(sys)

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
    with pytest.raises(TypeError):
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

def test_time_dependent_system_with_field():
    # good construction
    sysField = TimeDependentSystemWithField(hamiltonianFieldGood,
            name="with field")
    liouvillian = sysField.liouvillian(timeField,
            timeField+dtField, fieldField, derivativeField)
    np.testing.assert_almost_equal(liouvillian, liouvillianField)
    # Liouvillian calling
    with pytest.raises(TypeError):
        sysField.liouvillian()
    with pytest.raises(TypeError):
        sysField.liouvillian(0.5)
    with pytest.raises(TypeError):
        sysField.liouvillian(0.5, 1.0, stateField, derivativeField)
    with pytest.raises(TypeError):
        sysField.liouvillian(1j, 1.0, fieldField, derivativeField)
    with pytest.raises(AssertionError):
        sysField.liouvillian(2.0, 1.0, fieldField, derivativeField)
    with pytest.raises(TypeError):
        sysField.liouvillian(1.0, 1.0, "field", derivativeField)
    with pytest.raises(TypeError):
        sysField.liouvillian(0.5, 1.0, fieldField, sysField)
    # bad construction
    with pytest.raises(AssertionError):
        TimeDependentSystemWithField(
                hamiltonianFieldBad)
    with pytest.raises(AssertionError):
        TimeDependentSystemWithField(
                hamiltonianFieldBad2)

def test_mean_field_system():
    # good construction
    sysField = TimeDependentSystemWithField(hamiltonianFieldGood,
            name="with field")
    meansysField = MeanFieldSystem([sysField],
                                   fieldEomGood,
                                   name="mean-field sys")
    assert callable(meansysField.field_eom)
    derivative = meansysField.field_eom(timeField, [stateField], fieldField)
    assert np.isclose(derivative, derivativeField)
    # bad construction
    with pytest.raises(AssertionError):
        MeanFieldSystem(sysField, fieldEomGood)
    with pytest.raises(AssertionError):
        MeanFieldSystem([sysField], fieldEomBad)
    with pytest.raises(AssertionError):
        MeanFieldSystem([sysField], fieldEomBad2)
    with pytest.raises(AssertionError):
        MeanFieldSystem(fieldEomGood, [sysField])
    tsys = TimeDependentSystem(lambda t: t * np.eye(2))
    with pytest.raises(AssertionError):
        MeanFieldSystem([tsys], fieldEomGood)
    with pytest.raises(AssertionError): 
        # empty system list
        meansysField = MeanFieldSystem([],
                                       fieldEomGood,
                                       name="mean-field sys")


def test_parameterized_system():
    # good construction
    def hamiltonianParameterizedGood(x, y, z):
        h = np.zeros((2,2), dtype='complex128')
        for var, var_name in zip([x,y,z], ["x", "y", "z"]):
            h += var*operators.sigma(var_name)
        return h
    
    hamiltonianParameterizedBad = operators.sigma("x")
    hamiltonianParameterizedBad2 = operators.sigma("x").flatten()

    dt = 0.01
    xv = [0.0, 0.1, 0.2, 0.3]
    yv = [0.0, 0.0, 0.0, 0.0]
    zv = [1.0, 0.9, 0.8, 0.7]
    params = list(zip(xv,yv,zv))

    param_sys = ParameterizedSystem(hamiltonianParameterizedGood)
    assert param_sys.number_of_parameters == 3

    assert isinstance(param_sys.liouvillian(0.0,0.1,1.0),np.ndarray)

    props=param_sys.get_propagators(dt, (xv,yv,zv))
    prop_derivs=param_sys.get_propagator_derivatives(dt, (xv,yv,zv))

    with pytest.raises(TypeError):
        ParameterizedSystem(hamiltonianParameterizedBad)
    with pytest.raises(TypeError):
        ParameterizedSystem(hamiltonianParameterizedBad2)
    with pytest.raises(TypeError):
        param_sys.liouvillian()
    with pytest.raises(TypeError):
        param_sys.liouvillian(0.1)
    with pytest.raises(TypeError):
        props[0]
    with pytest.raises(TypeError):
        prop_derivs[0]

    assert isinstance(param_sys.hamiltonian,Callable)
    assert isinstance(param_sys.lindblad_operators,list)
    assert isinstance(param_sys.gammas,list)

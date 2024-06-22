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
Tests for the TempoParameters module.
"""

import pytest
import numpy as np

import oqupy as tempo

def test_tempo_parameters():
    tempo_param = tempo.TempoParameters(
        0.1, 1.0e-5, None, None, None, None, 2.0e-5, "rough", "bla")
    str(tempo_param)
    assert tempo_param.dt == 0.1
    assert tempo_param.epsrel == 1.0e-5
    assert tempo_param.tcut == None
    assert tempo_param.dkmax == None
    assert tempo_param.subdiv_limit == None
    assert tempo_param.liouvillian_epsrel == 2.0e-5
    with pytest.raises(AttributeError):
        tempo_param.dt = 0.05
    with pytest.raises(AttributeError):
        del tempo_param.dt
    with pytest.raises(AttributeError):
        tempo_param.tcut = 1
    with pytest.raises(AttributeError):
        del tempo_param.tcut
    with pytest.raises(AttributeError):
        tempo_param.epsrel = 1.0e-6
    with pytest.raises(AttributeError):
        del tempo_param.epsrel
    with pytest.raises(AttributeError):
        tempo_param.subdiv_limit = 256
    with pytest.raises(AttributeError):
        del tempo_param.subdiv_limit
    with pytest.raises(AttributeError):
        tempo_param.liouvillian_epsrel = 2.0e-6
    with pytest.raises(AttributeError):
        del tempo_param.liouvillian_epsrel

def test_tempo_parameters_bad_input():
    with pytest.raises(TypeError):
        tempo.TempoParameters("x", 1.0e-5, 4.2, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(-0.1, 1.0e-05, None, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, "x", 4.2, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, -1.0e-10, 4.2, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, "x", None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, 1.0e-05, -.5, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05,  None, "x",  None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, 1.0e-05, None, -5, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(AssertionError):
        tempo.TempoParameters(0.1, 1.0e-05, 1.0, 1, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, "x", None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, -99, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None,  "x", 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None, -1000, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None, None, "x", "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None, None, -2.0e-6, "rough", "bla")


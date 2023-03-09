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
Tests for the TempoParameters module.
"""

import pytest
import numpy as np

import oqupy as tempo

def test_tempo_parameters():
    tempo_param = tempo.TempoParameters(
        0.1, None, 1.0e-5, None, None, 2.0e-5, "rough", "bla")
    str(tempo_param)
    assert tempo_param.dt == 0.1
    assert tempo_param.tcut == None
    assert tempo_param.dkmax == None
    assert tempo_param.epsrel == 1.0e-5
    assert tempo_param.subdiv_limit == None
    assert tempo_param.liouvillian_epsrel == 2.0e-5
    tempo_param.dt = 0.05
    tempo_param.tcut = 42
    tempo_param.epsrel = 1.0e-6
    tempo_param.subdiv_limit = 256
    tempo_param.liouvillian_epsrel = 2.0e-6
    assert tempo_param.dt == 0.05
    assert tempo_param.tcut == 42
    assert tempo_param.dkmax == 42
    assert tempo_param.epsrel == 1.0e-6
    assert tempo_param.liouvillian_epsrel == 2.0e-6
    tempo_param.tcut = 0.5
    assert tempo_param.tcut == 0.5
    assert tempo_param.dkmax == 10
    del tempo_param.tcut
    assert tempo_param.tcut == None
    assert tempo_param.dkmax == None
    del tempo_param.subdiv_limit
    assert tempo_param.subdiv_limit == None

def test_tempo_parameters_bad_input():
    with pytest.raises(TypeError):
        tempo.TempoParameters("x", 42, 1.0e-5, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, "x", 1.0e-5, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(ValueError):
        tempo.TempoParameters(0.1, -1, 1.0e-05, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 42, "x", None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 42, 1.0e-05, "x", None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 42, 1.0e-05, None, "x", 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 42, 1.0e-05, None, None, "x", "rough", "bla")


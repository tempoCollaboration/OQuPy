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

def test_gibbs_parameters():
    gibbs_param = tempo.GibbsParameters(0.1, 22, 1.0e-5, "rough", "bla")
    str(gibbs_param)
    assert gibbs_param.temperature == 0.1
    assert gibbs_param.n_steps == 22
    assert gibbs_param.epsrel == 1.0e-5

    with pytest.raises(AttributeError):
        gibbs_param.temperature = 0.05
    with pytest.raises(AttributeError):
        del gibbs_param.temperature
    with pytest.raises(AttributeError):
        gibbs_param.n_steps = 6
    with pytest.raises(AttributeError):
        del gibbs_param.nsteps
    with pytest.raises(AttributeError):
        gibbs_param.epsrel = 1.0e-6
    with pytest.raises(AttributeError):
        del gibbs_param.epsrel
    with pytest.raises(AttributeError):
        gibbs_param.subdiv_limit = 256
    with pytest.raises(AttributeError):
        del gibbs_param.subdiv_limit
    with pytest.raises(AttributeError):
        gibbs_param.liouvillian_epsrel = 2.0e-6
    with pytest.raises(AttributeError):
        del gibbs_param.liouvillian_epsrel

def test_tempo_parameters_bad_input():
    with pytest.raises(TypeError):
        tempo.TempoParameters("x", 1.0e-5, 4.2, None, None, None, 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, "x", 4.2, None, None, None, 2.0e-6, "rough", "bla")
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
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None,  "x", 2.0e-6, "rough", "bla")
    with pytest.raises(TypeError):
        tempo.TempoParameters(0.1, 1.0e-05, 4.2, None, None, None, "x", "rough", "bla")


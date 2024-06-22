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


from oqupy import tempo

def test_gibbs_parameters():
    temperature = 0.1
    n_steps = 22
    epsrel = 1.0e-5
    gibbs_param = tempo.GibbsParameters(n_steps, epsrel, "rough", "bla")
    str(gibbs_param)

    properties = ['n_steps', 'epsrel']
    values = [n_steps, epsrel]

    for p, v in zip(properties, values):
        assert getattr(gibbs_param, p) == v
        with pytest.raises(AttributeError):
            setattr(gibbs_param, p, v + 1)
        with pytest.raises(AttributeError):
            delattr(gibbs_param, p)

        

def test_gibbs_parameters_bad_input():
    keys = ['n_steps', 'epsrel']
    good_inputs = [10, 2.0e-6]
    bad_inputs = [1, -2.0e-6]
    for k, b in zip(keys, bad_inputs):
        input = dict(zip(keys, good_inputs))
        with pytest.raises(TypeError):
            input[k] = 'x'
            tempo.GibbsParameters(**input, name="rough", description="bla")
        with pytest.raises(ValueError):
            input[k] = b
            tempo.GibbsParameters(**input, name="rough", description="bla")

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
Tests for the time_evovling_mpo.bath module.
"""

import pytest

import numpy as np

from time_evolving_mpo.bath import Bath
from time_evolving_mpo.spectral_density import StandardSD
from time_evolving_mpo import operators

def test_bath():
    wc = 4.0
    alpha = 0.3
    temperature = 0.1
    spectral_density = StandardSD(alpha=alpha,
                                  zeta=1.0,
                                  cutoff=wc,
                                  cutoff_type="exponential")
    coupling_operator = operators.pauli("z")
    name = "ohmic"
    description = """ Ohmic spectral density. \n J(w) = 2 alpha w exp(-w/wc) """
    description_dict = {"alpha": alpha, "wc": wc}

    # try a minimal example
    bath_A = Bath(coupling_operator, spectral_density, temperature)
    bath_A.name = name
    bath_A.descripton = description
    bath_A.description_dict = description_dict

    # see if the properties work
    str(bath_A)
    np.testing.assert_equal(bath_A.coupling_operator, coupling_operator)
    assert bath_A.dimension == 2
    assert bath_A.spectral_density.zeta == 1.0
    assert bath_A.temperature == temperature
    del bath_A.name
    del bath_A.description
    del bath_A.description_dict
    str(bath_A)
    with pytest.raises(AssertionError):
        bath_A.name = 0.1
    with pytest.raises(AssertionError):
        bath_A.description = ["bla"]
    with pytest.raises(AssertionError):
        bath_A.description_dict = ["bla"]

    # try a full example
    Bath(coupling_operator,
         spectral_density,
         temperature,
         name=name,
         description=description,
         description_dict=description_dict)

    # try bad examples
    with pytest.raises(AssertionError):
        coupling_op = "bla"
        Bath(coupling_op, spectral_density)
    with pytest.raises(AssertionError):
        coupling_op = np.array([1.0,0.0])
        Bath(coupling_op, spectral_density)
    with pytest.raises(AssertionError):
        coupling_op = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
        Bath(coupling_op, spectral_density)
    with pytest.raises(NotImplementedError):
        coupling_op = np.array([[1.0,0.1],[0.1,1.0]])
        Bath(coupling_op, spectral_density)
    with pytest.raises(AssertionError):
        Bath(coupling_operator, spectral_density="bla")
    with pytest.raises(AssertionError):
        Bath(coupling_operator, spectral_density, temperature="bla")
    with pytest.raises(ValueError):
        Bath(coupling_operator, spectral_density, temperature=-0.2)

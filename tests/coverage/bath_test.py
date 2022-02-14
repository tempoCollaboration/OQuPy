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
Tests for the time_evovling_mpo.bath module.
"""

import pytest

import numpy as np

from oqupy.bath import Bath
from oqupy.correlations import PowerLawSD
from oqupy import operators

def test_bath():
    wc = 4.0
    alpha = 0.3
    temperature = 0.1
    correlations = PowerLawSD(alpha=alpha,
                              zeta=1.0,
                              cutoff=wc,
                              cutoff_type="exponential",
                              temperature=temperature)
    coupling_operator = np.array([[1,0],[0,-1]])
    name = "phonon bath"
    description = """ Ohmic spectral density. \n J(w) = 2 alpha w exp(-w/wc) """
    description_dict = {"alpha": alpha, "wc": wc, "T": temperature}

    # try a minimal example
    bath_A = Bath(coupling_operator, correlations)
    bath_A.name = name
    bath_A.descripton = description
    bath_A.description_dict = description_dict

    # see if the properties work
    str(bath_A)
    np.testing.assert_equal(bath_A.coupling_operator, coupling_operator)
    assert bath_A.dimension == 2
    assert bath_A.correlations.zeta == 1.0
    del bath_A.name
    del bath_A.description
    del bath_A.description_dict
    str(bath_A)
    with pytest.raises(AssertionError):
        bath_A.name = 0.1
    with pytest.raises(AssertionError):
        bath_A.description = ["bla"]

    # try a full example
    Bath(coupling_operator,
         correlations,
         name=name,
         description=description)

    # try non-diagonal coupling
    coupling_op = np.array([[0.0,-1.0j],[1.0j,0.0]])
    bath_B = Bath(coupling_op, correlations)
    u = bath_B.unitary_transform
    op = bath_B.coupling_operator
    assert np.allclose(coupling_op, \
            u @ op @ u.conjugate().T)

def test_bath_bad_input():
    wc = 4.0
    alpha = 0.3
    temperature = 0.1
    correlations = PowerLawSD(alpha=alpha,
                              zeta=1.0,
                              cutoff=wc,
                              cutoff_type="exponential",
                              temperature=temperature)
    with pytest.raises(AssertionError):
        coupling_op = "bla"
        Bath(coupling_op, correlations)
    with pytest.raises(AssertionError):
        coupling_op = np.array([1.0,0.0])
        Bath(coupling_op, correlations)
    with pytest.raises(AssertionError):
        coupling_op = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
        Bath(coupling_op, correlations)
    with pytest.raises(AssertionError):
        coupling_op = np.array([[0.0,-1.0j],[1.0j,0.0]])
        Bath(coupling_op, correlations="bla")

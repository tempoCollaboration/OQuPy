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
Tests for the time_evovling_mpo.spectral_density module.
"""

import pytest

from time_evolving_mpo.spectral_density import BaseSD
from time_evolving_mpo import CustomFunctionSD
from time_evolving_mpo import CustomDataSD
from time_evolving_mpo import StandardSD
from time_evolving_mpo import OhmicSD
from time_evolving_mpo import LorentzSD


FOO = 0.0


def test_base_sd():
    base_sd=BaseSD()
    with pytest.raises(NotImplementedError):
        base_sd.correlation(FOO, FOO)
    with pytest.raises(NotImplementedError):
        base_sd.correlation_2d_integral(FOO, FOO, FOO, FOO)

def test_custom_function_sd():
    CustomFunctionSD(FOO, FOO, FOO)

def test_custom_data_sd():
    CustomDataSD(FOO, FOO)

def test_standard_sd():
    StandardSD(FOO, FOO, FOO, FOO)

def test_ohmic_sd():
    OhmicSD(FOO, FOO, FOO)

def test_lorentz_sd():
    LorentzSD(FOO, FOO, FOO)

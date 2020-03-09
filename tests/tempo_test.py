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
Tests for the time_evovling_mpo.tempo module.
"""

import pytest

from time_evolving_mpo import Tempo
from time_evolving_mpo import TempoParameters
from time_evolving_mpo import guess_tempo_parameters

def test_tempo():
    tempo_A=Tempo()
    tempo_A.check_convergence()
    tempo_A.compute()
    tempo_A.get_dynamics()

def test_tempo_parameters():
    TempoParameters()

def test_guess_tempo_parameters():
    res = guess_tempo_parameters()
    assert isinstance(res,TempoParameters)

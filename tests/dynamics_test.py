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
Tests for the time_evovling_mpo.dynamics module.
"""

import pytest

from time_evolving_mpo import Dynamics
from time_evolving_mpo import distance
from time_evolving_mpo import norms


def test_dynamics():
    dynamics_A = Dynamics()
    dynamics_A.export()
    dynamics_A.get_expectations()

def test_distance():
    distance()

def test_norms():
    norms()

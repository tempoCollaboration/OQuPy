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
Tests for the oqupy.tempo.backends.base_backends module.
"""
import pytest

import numpy as np
import tensornetwork as tn

import oqupy as tempo
import oqupy.tempo.backends.base_backends as bb
pass

def test_base_tempo_backend():
    tempo_back = bb.BaseTempoBackend(
                    initial_state=None,
                    influence=None,
                    unitary_transform=None,
                    propagators=None,
                    sum_north=None,
                    sum_west=None,
                    dkmax=None,
                    epsrel=None,
                    config=None)
    with pytest.raises(NotImplementedError):
        tempo_back.initialize()
    with pytest.raises(NotImplementedError):
        tempo_back.compute_step()


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
Factory for backends.
"""

from typing import Dict, Optional, Text

from oqupy.config import TEMPO_BACKEND, TEMPO_BACKEND_CONFIG
from oqupy.backends.base_backends import BaseTempoBackend
from oqupy.backends.tensor_network.tempo_backend \
    import TensorNetworkTempoBackend
from oqupy.config import PT_TEMPO_BACKEND, PT_TEMPO_BACKEND_CONFIG
from oqupy.backends.base_backends import BasePtTempoBackend
from oqupy.backends.tensor_network.pt_tempo_backend \
    import TensorNetworkPtTempoBackend

TEMPO_BACKEND_DICT = {
    'tensor-network': TensorNetworkTempoBackend,
    }
PT_TEMPO_BACKEND_DICT = {
    'tensor-network': TensorNetworkPtTempoBackend,
    }

def get_tempo_backend(
        name: Optional[Text] = None,
        config: Optional[Dict] = None) -> BaseTempoBackend:
    """Returns a backend class and configuration. """
    # input checks
    if name is None:
        name = TEMPO_BACKEND
    assert name in TEMPO_BACKEND_DICT, \
        f"Unknown tempo backend '{name}'. " \
        + f"Known backends: {TEMPO_BACKEND_DICT.keys()}."

    if config is None:
        config = TEMPO_BACKEND_CONFIG[name]

    return TEMPO_BACKEND_DICT[name], config

def get_pt_tempo_backend(
        name: Optional[Text] = None,
        config: Optional[Dict] = None) -> BasePtTempoBackend:
    """Returns a backend class and configuration. """
    # input checks
    if name is None:
        name = PT_TEMPO_BACKEND
    assert name in PT_TEMPO_BACKEND_DICT, \
        f"Unknown process tensor tempo backend '{name}'. " \
        + f"Known backends: {PT_TEMPO_BACKEND_DICT.keys()}."

    if config is None:
        config = PT_TEMPO_BACKEND_CONFIG[name]

    return PT_TEMPO_BACKEND_DICT[name], config

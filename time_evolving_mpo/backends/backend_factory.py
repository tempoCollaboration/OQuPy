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

from time_evolving_mpo.config import TEMPO_BACKEND, TEMPO_BACKEND_CONFIG
from time_evolving_mpo.backends.base_backends import BaseTempoBackend
from time_evolving_mpo.backends.tensor_network.tensor_network_backend \
    import TensorNetworkTempoBackend


TEMPO_BACKEND_DICT = {
    'tensor-network': TensorNetworkTempoBackend,
    }


def get_tempo_backend(
        name: Optional[Text] = None,
        config: Optional[Dict] = None) -> BaseTempoBackend:
    """Returns a backend object. """
    # input checks
    if name is None:
        name = TEMPO_BACKEND
    assert name in TEMPO_BACKEND_DICT, \
        f"Unknown tempo backend '{name}'. " \
        + f"Known backends: {TEMPO_BACKEND_DICT.keys()}."

    if config is None:
        config = TEMPO_BACKEND_CONFIG[name]

    return TEMPO_BACKEND_DICT[name], config

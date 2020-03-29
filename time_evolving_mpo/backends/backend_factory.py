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

from time_evolving_mpo.config import BACKEND, BACKEND_CONFIG
from time_evolving_mpo.backends.base_backend \
    import BaseBackend
from time_evolving_mpo.backends.example.example_backend \
    import ExampleBackend


BACKEND_DICT = {
    'example': ExampleBackend,
    }


def get_backend(
        name: Optional[Text] = None,
        config: Optional[Dict] = None) -> BaseBackend:
    """Returns a backend object. """
    # input checks
    if name is None:
        name = BACKEND
    assert name in BACKEND_DICT, \
        "Unknown backend '{}'. Known backends: {}.".format(name,
                                                           BACKEND_DICT.keys())
    if config is None:
        config = BACKEND_CONFIG[name]

    return BACKEND_DICT[name](config)

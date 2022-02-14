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
Tests for the time_evovling_mpo.base_api module.
"""

import pytest

from oqupy.base_api import BaseAPIClass


def test_base_api_class():
    name = "Some name"
    description = """ Description bla bla. \n J(w) = 2 alpha w exp(-w/wc) """

    base_api = BaseAPIClass(name=name,
                            description=description)
    base_api = BaseAPIClass()
    base_api.name = name
    base_api.descripton = description

    str(base_api)
    del base_api.name
    del base_api.description
    str(base_api)
    with pytest.raises(AssertionError):
        base_api.name = 0.1
    with pytest.raises(AssertionError):
        base_api.description = ["bla"]

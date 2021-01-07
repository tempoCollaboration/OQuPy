# Copyright 2021 The TEMPO Collaboration
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

from time_evolving_mpo.base_api import BaseAPIClass


def test_base_api_class():
    name = "Some name"
    description = """ Description bla bla. \n J(w) = 2 alpha w exp(-w/wc) """
    description_dict = {"alpha": 0.1, "wc": 4.0}

    base_api = BaseAPIClass(name=name,
                            description=description,
                            description_dict=description_dict)
    base_api = BaseAPIClass()
    base_api.name = name
    base_api.descripton = description
    base_api.description_dict = description_dict

    str(base_api)
    print(base_api.description_dict)
    del base_api.name
    del base_api.description
    del base_api.description_dict
    str(base_api)
    with pytest.raises(AssertionError):
        base_api.name = 0.1
    with pytest.raises(AssertionError):
        base_api.description = ["bla"]
    with pytest.raises(AssertionError):
        base_api.description_dict = ["bla"]

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
Tests for the time_evovling_mpo.file_formats module.
"""

import pytest

from time_evolving_mpo.file_formats import check_tempo_dynamics_file


GOOD_FILE_DYNAMICS_V1 = "tests/data/test_v1_0_good_file.tempoDynamics"
BAD_FILE_DYNAMICS_V1 = "tests/data/test_v1_0_bad_file.tempoDynamics"


def test_check_tempo_dynamics_file_good():
    assert check_tempo_dynamics_file(GOOD_FILE_DYNAMICS_V1)

def test_check_tempo_dynamics_file_bad():
    assert check_tempo_dynamics_file(BAD_FILE_DYNAMICS_V1) is False

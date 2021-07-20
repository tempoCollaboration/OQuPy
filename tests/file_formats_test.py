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
Tests for the time_evovling_mpo.file_formats module.
"""

import pytest
import os, contextlib

from time_evolving_mpo.file_formats import check_tempo_dynamics_file
from time_evolving_mpo.file_formats import print_tempo_dynamics_file

# -----------------------------------------------------------------------------

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper

# -----------------------------------------------------------------------------

BAD_FILES_TEMPO_DYNAMICS_V1 = [
    "tests/data/test_v1_0_bad_file_A.tempoDynamics",
    "tests/data/test_v1_0_bad_file_B.tempoDynamics",
    "tests/data/test_v1_0_bad_file_C.tempoDynamics",
    ]
GOOD_FILES_TEMPO_DYNAMICS_V1 = [
    "tests/data/test_v1_0_good_file_A.tempoDynamics",
    "tests/data/test_v1_0_good_file_B.tempoDynamics",
    "tests/data/test_v1_0_good_file_C.tempoDynamics",
    ]

# -----------------------------------------------------------------------------

def test_check_tempo_dynamics_files_good():
    for filename in GOOD_FILES_TEMPO_DYNAMICS_V1:
        assert check_tempo_dynamics_file(filename)

def test_check_tempo_dynamics_files_bad():
    for filename in BAD_FILES_TEMPO_DYNAMICS_V1:
        assert check_tempo_dynamics_file(filename) is False

@supress_stdout
def test_print_tempo_dynamics_files():
    for filename in GOOD_FILES_TEMPO_DYNAMICS_V1:
        print_tempo_dynamics_file(filename)

# -----------------------------------------------------------------------------

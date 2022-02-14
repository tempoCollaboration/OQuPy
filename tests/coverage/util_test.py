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
Tests for the time_evovling_mpo.util module.
"""

import pytest

import numpy as np

from oqupy.util import BaseProgress, ProgressBar


def test_base_progress():
    prog = BaseProgress()
    with pytest.raises(NotImplementedError):
        prog.__enter__()
    with pytest.raises(NotImplementedError):
        prog.__exit__(None, None, None)
    with pytest.raises(NotImplementedError):
        prog.update(4)

def test_progess_bar():
    with ProgressBar(10) as prog_bar:
        prog_bar.update(None)
    with ProgressBar(0) as prog_bar:
        prog_bar.update(None)

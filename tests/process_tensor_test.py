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
Tests for the time_evovling_mpo.process_tensor module.
"""

import pytest

from time_evolving_mpo import ProcessTensor
from time_evolving_mpo import apply_control_to_process_tensor
from time_evolving_mpo import apply_system_to_process_tensor
from time_evolving_mpo import compute_process_tensor
from time_evolving_mpo import ProcessTensorParameters
from time_evolving_mpo import guess_process_tensor_parameters


def test_process_tensor():
    process_tensor_A=ProcessTensor()
    process_tensor_A.check_convergence()
    process_tensor_A.export()

def test_apply_control_to_process_tensor():
    apply_control_to_process_tensor()

def test_apply_system_to_process_tensor():
    apply_system_to_process_tensor()

def test_compute_process_tensor():
    compute_process_tensor()

def test_process_tensor_parameters():
    ProcessTensorParameters()

def test_guess_process_tensor_parameters():
    res = guess_process_tensor_parameters()
    assert isinstance(res,ProcessTensorParameters)

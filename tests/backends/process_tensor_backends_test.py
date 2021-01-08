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
Tests for the process_tensor backends.
"""
import pytest

import numpy as np
import tensornetwork as tn

import time_evolving_mpo as tempo


TEMP_FILE = "tests/data/temp.processTensor"


@pytest.mark.parametrize('backend,backend_config',
                        [("tensor-network", {"backend":"numpy"}),
                         ("tensor-network", None)])
def test_process_tensor_backends(backend, backend_config):
    initial_state = tempo.operators.spin_dm("y+")
    initial_tensor = initial_state.flatten().reshape((1,-1))
    system = tempo.System(tempo.operators.sigma("z"))

    pt = tempo.ProcessTensor(
        times = None,
        tensors = [],
        initial_tensor = None,
        backend=backend,
        backend_config=backend_config)
    pt.export(TEMP_FILE,overwrite=True)
    pt.get_bond_dimensions()

    pt = tempo.ProcessTensor(
        times = None,
        tensors = [],
        initial_tensor = initial_tensor,
        backend=backend,
        backend_config=backend_config)
    pt.export(TEMP_FILE,overwrite=True)
    pt.get_bond_dimensions()
    pt.compute_final_state_from_system(system=system)
    pt.compute_dynamics_from_system(system=system)

    tensor = tempo.operators.identity(4).reshape((1,1,4,4))
    pt = tempo.ProcessTensor(
        times = None,
        tensors = [tensor],
        initial_tensor = initial_tensor,
        backend=backend,
        backend_config=backend_config)
    pt.export(TEMP_FILE,overwrite=True)
    pt.get_bond_dimensions()
    pt.compute_final_state_from_system(system=system)
    pt.compute_dynamics_from_system(system=system)

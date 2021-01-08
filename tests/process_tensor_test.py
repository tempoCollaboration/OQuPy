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
Tests for the time_evovling_mpo.process_tensor module.
"""

import pytest

import time_evolving_mpo as tempo


TEMP_FILE = "tests/data/temp.processTensor"

# -- prepare a process tensor -------------------------------------------------

system = tempo.System(tempo.operators.sigma("x"))
initial_state = tempo.operators.spin_dm("z+")
correlations = tempo.PowerLawSD(alpha=0.3,
                                zeta=1.0,
                                cutoff=5.0,
                                cutoff_type="exponential",
                                temperature=0.2,
                                name="ohmic")
bath = tempo.Bath(0.5*tempo.operators.sigma("z"),
                    correlations,
                    name="phonon bath")
tempo_params = tempo.PtTempoParameters(dt=0.1,
                                         dkmax=5,
                                         epsrel=10**(-5))
pt = tempo.pt_tempo_compute(bath,
                            start_time=0.0,
                            end_time=1.0,
                            parameters=tempo_params)
pt.export(TEMP_FILE, overwrite=True)
del pt


def test_process_tensor():
    pt = tempo.import_process_tensor(TEMP_FILE)
    str(pt)
    pt.times
    pt.get_bond_dimensions()
    pt.compute_final_state_from_system(system, initial_state)
    pt.compute_dynamics_from_system(system, initial_state)
    with pytest.raises(FileExistsError):
        pt.export(TEMP_FILE)

    with pytest.raises(AssertionError):
        pt.compute_final_state_from_system(system)
    with pytest.raises(AssertionError):
        pt.compute_dynamics_from_system(system)
    with pytest.raises(AssertionError):
        pt.compute_final_state_from_system(system,"bla")
    with pytest.raises(AssertionError):
        pt.compute_dynamics_from_system(system,"bla")

# -----------------------------------------------------------------------------

GOOD_FILES_PROCESS_TENSOR_V1 = [
    "tests/data/test_v1_0_good_file_A.processTensor",
    "tests/data/test_v1_0_good_file_B.processTensor",
    "tests/data/test_v1_0_good_file_C.processTensor",
    "tests/data/test_v1_0_good_file_D.processTensor",
    ]

def test_import_process_tensor():
    for filename in GOOD_FILES_PROCESS_TENSOR_V1:
        tempo.import_process_tensor(filename)

def test_process_tensor_bad_input():
    tempo.ProcessTensor(times=None,
                        tensors=[])

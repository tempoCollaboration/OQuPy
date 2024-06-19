# Copyright 2022 The oqupy Collaboration
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

import oqupy

TEMP_FILE_1 = "./tests/data/temp/temp1.hdf5"
TEMP_FILE_2 = "./tests/data/temp/temp2.hdf5"

# -- prepare a process tensor -------------------------------------------------

system = oqupy.System(oqupy.operators.sigma("x"))
initial_state = oqupy.operators.spin_dm("z+")
correlations = oqupy.PowerLawSD(
    alpha=0.3,
    zeta=1.0,
    cutoff=5.0,
    cutoff_type="exponential",
    temperature=0.2,
    name="ohmic")
bath1 = oqupy.Bath(
    0.5*oqupy.operators.sigma("z"),
    correlations,
    name="phonon bath")
bath2 = oqupy.Bath(
    0.5*oqupy.operators.sigma("x"),
    correlations,
    name="phonon bath")
tempo_params = oqupy.TempoParameters(
    dt=0.1,
    tcut=0.5,
    epsrel=10**(-5))
pt1 = oqupy.pt_tempo_compute(
    bath1,
    start_time=0.0,
    end_time=0.3,
    parameters=tempo_params)
pt2 = oqupy.pt_tempo_compute(
    bath2,
    start_time=0.0,
    end_time=0.3,
    parameters=tempo_params)
pt1.name = "PT1"
pt2.name = "PT2"
pt1.export(TEMP_FILE_1, overwrite=True)
pt2.export(TEMP_FILE_2, overwrite=True)
del pt1
del pt2


def test_process_tensor():
    pt1 = oqupy.import_process_tensor(TEMP_FILE_1, process_tensor_type="simple")
    str(pt1)
    pt1.get_bond_dimensions()
    with pytest.raises(OSError):
        pt1.export(TEMP_FILE_1)

    pt2 = oqupy.import_process_tensor(TEMP_FILE_2, process_tensor_type="file")
    str(pt2)
    assert pt2.name == "PT2"
    pt2.get_bond_dimensions()

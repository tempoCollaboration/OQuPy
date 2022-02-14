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
Tests for the time_evovling_mpo.process_tensor module.
"""

import pytest

import oqupy as tempo


TEMP_FILE = "tests/data/temp.processTensor"

# -- prepare a process tensor -------------------------------------------------

system = tempo.System(tempo.operators.sigma("x"))
initial_state = tempo.operators.spin_dm("z+")
correlations = tempo.PowerLawSD(
    alpha=0.3,
    zeta=1.0,
    cutoff=5.0,
    cutoff_type="exponential",
    temperature=0.2,
    name="ohmic")
bath = tempo.Bath(
    0.5*tempo.operators.sigma("z"),
    correlations,
    name="phonon bath")
tempo_params = tempo.TempoParameters(
    dt=0.1,
    dkmax=5,
    epsrel=10**(-5))
pt = tempo.pt_tempo_compute(
    bath,
    start_time=0.0,
    end_time=1.0,
    parameters=tempo_params)
pt.export(TEMP_FILE, overwrite=True)
del pt


def test_process_tensor():
    pt = tempo.import_process_tensor(TEMP_FILE, process_tensor_type="simple")
    str(pt)
    pt.get_bond_dimensions()
    with pytest.raises(OSError):
        pt.export(TEMP_FILE)


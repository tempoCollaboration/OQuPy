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
Tests for the time_evovling_mpo.pt_tebd module.
"""

import pytest

import numpy as np
import oqupy

up_dm = oqupy.operators.spin_dm("z+")
system_chain = oqupy.SystemChain(hilbert_space_dimensions=[2,3])
initial_augmented_mps = oqupy.AugmentedMPS([up_dm, np.diag([1,0,0])])
pt_tebd_params = oqupy.PtTebdParameters(dt=0.2, order=2, epsrel=1.0e-4)

def test_get_augmented_mps():
    pt_tebd = oqupy.PtTebd(
        initial_augmented_mps=initial_augmented_mps,
        system_chain=system_chain,
        process_tensors=[None, None],
        parameters=pt_tebd_params)

    augmented_mps = pt_tebd.get_augmented_mps()
    assert augmented_mps.gammas[0].shape == (1,4,1,1)
    assert augmented_mps.gammas[1].shape == (1,9,1,1)

    pt_tebd.compute(end_step=2, progress_type='silent')
    augmented_mps = pt_tebd.get_augmented_mps()
    assert augmented_mps.gammas[0].shape == (1,4,1,1)
    assert augmented_mps.gammas[1].shape == (1,9,1,1)

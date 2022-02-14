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
Test for the time_evovling_mpo.process_tensor module multiple environment
functionality.
"""

import numpy as np

import oqupy


# -- prepare process tensors -------------------------------------------------

system = oqupy.System(oqupy.operators.sigma("x"))
initial_state = oqupy.operators.spin_dm("z+")
correlations = oqupy.PowerLawSD(alpha=0.05,
                                zeta=1.0,
                                cutoff=5.0,
                                cutoff_type="exponential",
                                temperature=0.2,
                                name="ohmic")
correlations2 = oqupy.PowerLawSD(alpha=0.1,
                                zeta=1.0,
                                cutoff=5.0,
                                cutoff_type="exponential",
                                temperature=0.2,
                                name="ohmic")
bath = oqupy.Bath(0.5*oqupy.operators.sigma("z"),
                    correlations,
                    name="phonon bath")
bath2 = oqupy.Bath(0.5*oqupy.operators.sigma("z"),
                    correlations2,
                    name="half-coupling phonon bath")
tempo_params = oqupy.TempoParameters(dt=0.1,
                                     dkmax=5,
                                     epsrel=10**(-6))
pt = oqupy.pt_tempo_compute(bath,
                            start_time=0.0,
                            end_time=1.0,
                            parameters=tempo_params)
pt2 = oqupy.pt_tempo_compute(bath2,
                            start_time=0.0,
                            end_time=1.0,
                            parameters=tempo_params)


def test_multi_env_dynamics():
    dyns = oqupy.compute_dynamics(system, [pt,pt],initial_state=initial_state)
    dyns2 = oqupy.compute_dynamics(system, [pt2],initial_state=initial_state)
    np.testing.assert_almost_equal(dyns.states,dyns2.states,decimal=5)

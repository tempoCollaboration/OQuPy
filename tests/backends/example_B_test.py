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
Tests for the time_evovling_mpo.backends.tensor_network modules.
"""

import pytest
import numpy as np

import oqupy as tempo

# -----------------------------------------------------------------------------
# -- Test B: non-diagonal coupling --------------------------------------------

@pytest.mark.parametrize('backend',["tensor-network"])
def test_tensor_network_tempo_backend_non_diag(backend):
    Omega = 1.0
    omega_cutoff = 5.0
    alpha = 0.3

    sx=tempo.operators.sigma("x")
    sy=tempo.operators.sigma("y")
    sz=tempo.operators.sigma("z")

    bases = [{"sys_op":sx, "coupling_op":sz, \
                "init_state":tempo.operators.spin_dm("y+")},
             {"sys_op":sy, "coupling_op":sx, \
                "init_state":tempo.operators.spin_dm("z+")},
             {"sys_op":sz, "coupling_op":sy, \
                "init_state":tempo.operators.spin_dm("x+")}]

    results = []
    for i, base in enumerate(bases):
        system = tempo.System(0.5*base["sys_op"])
        correlations = tempo.PowerLawSD(alpha=alpha,
                                        zeta=1,
                                        cutoff=omega_cutoff,
                                        cutoff_type='exponential',
                                        max_correlation_time=8.0)
        bath = tempo.Bath(0.5*base["coupling_op"], correlations)
        tempo_parameters = tempo.TempoParameters(dt=0.1,
                                                 dkmax=30,
                                                 epsrel=10**(-5))

        dynamics = tempo.tempo_compute(system=system,
                                       bath=bath,
                                       initial_state=base["init_state"],
                                       start_time=0.0,
                                       end_time=1.0,
                                       parameters=tempo_parameters,
                                       backend=backend)

        _, s_x = dynamics.expectations(0.5*tempo.operators.sigma("x"),
                                       real=True)
        _, s_y = dynamics.expectations(0.5*tempo.operators.sigma("y"),
                                       real=True)
        _, s_z = dynamics.expectations(0.5*tempo.operators.sigma("z"),
                                       real=True)
        if i == 0:
            results.append(np.array([s_x, s_y, s_z]))
        elif i == 1:
            results.append(np.array([s_y, s_z, s_x]))
        elif i == 2:
            results.append(np.array([s_z, s_x, s_y]))

    assert np.allclose(results[0], results[1], atol=tempo_parameters.epsrel)
    assert np.allclose(results[0], results[2], atol=tempo_parameters.epsrel)


@pytest.mark.parametrize('backend',["tensor-network"])
def test_tensor_network_pt_tempo_backend_non_diag(backend):
    Omega = 1.0
    omega_cutoff = 5.0
    alpha = 0.3

    sx=tempo.operators.sigma("x")
    sy=tempo.operators.sigma("y")
    sz=tempo.operators.sigma("z")

    bases = [{"sys_op":sx, "coupling_op":sz, \
                "init_state":tempo.operators.spin_dm("y+")},
             {"sys_op":sy, "coupling_op":sx, \
                "init_state":tempo.operators.spin_dm("z+")},
             {"sys_op":sz, "coupling_op":sy, \
                "init_state":tempo.operators.spin_dm("x+")}]

    results = []
    for i, base in enumerate(bases):
        system = tempo.System(0.5*base["sys_op"])
        correlations = tempo.PowerLawSD(alpha=alpha,
                                        zeta=1,
                                        cutoff=omega_cutoff,
                                        cutoff_type='exponential',
                                        max_correlation_time=8.0)
        bath = tempo.Bath(0.5*base["coupling_op"], correlations)
        tempo_parameters = tempo.PtTempoParameters(dt=0.1,
                                                 dkmax=30,
                                                 epsrel=10**(-5))

        pt = tempo.pt_tempo_compute(bath=bath,
                                    start_time=0.0,
                                    end_time=1.0,
                                    parameters=tempo_parameters,
                                    backend=backend)
        dynamics = pt.compute_dynamics_from_system(
                    system=system,
                    initial_state=base["init_state"])
        _, s_x = dynamics.expectations(0.5*tempo.operators.sigma("x"),
                                       real=True)
        _, s_y = dynamics.expectations(0.5*tempo.operators.sigma("y"),
                                       real=True)
        _, s_z = dynamics.expectations(0.5*tempo.operators.sigma("z"),
                                       real=True)
        if i == 0:
            results.append(np.array([s_x, s_y, s_z]))
        elif i == 1:
            results.append(np.array([s_y, s_z, s_x]))
        elif i == 2:
            results.append(np.array([s_z, s_x, s_y]))

    assert np.allclose(results[0], results[1], atol=tempo_parameters.epsrel)
    assert np.allclose(results[0], results[2], atol=tempo_parameters.epsrel)

# -----------------------------------------------------------------------------

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
Performance test for degeneracy checking in PT-TEMPO generation.
"""
import sys
from time import perf_counter
sys.path.insert(0,'.')

import numpy as np

import oqupy


def pt_degen_performance_A(spin_size, unique):
    p = {'alpha': 0.1,
         'zeta': 1.0,
         'temperature': 0.0,
         'cutoff': 1.0,
         'cutoff_type': 'exponential',
         'dt': 0.2,
         'epsrel': 1e-6,
         'tcut': 2.0,
         'end_t': 2.0}

    correlations = oqupy.PowerLawSD(alpha=p['alpha'],
                                    zeta=p['zeta'],
                                    cutoff=p['cutoff'],
                                    cutoff_type=p['cutoff_type'],
                                    temperature=p['temperature'])

    coupling_op = np.diag(
        np.arange(spin_size, -spin_size+0.1, -1, dtype=float))

    bath = oqupy.Bath(coupling_operator=coupling_op,
                      correlations=correlations)
    pt_tempo_parameters = oqupy.TempoParameters(
        dt=p['dt'], tcut=p['tcut'], epsrel=p['epsrel'])
    result = {}
    t0 = perf_counter()
    oqupy.pt_tempo_compute(bath=bath,
                           start_time=0.0,
                           end_time=p['end_t'],
                           parameters=pt_tempo_parameters,
                           unique=unique)
    result['walltime'] = perf_counter()-t0
    result['unique'] = unique
    result['spin_size'] = spin_size
    return result


parameters_unique = [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                     [True]]
parameters_non_unique = [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                         [False]]

ALL_TESTS = [(pt_degen_performance_A, [parameters_unique, parameters_non_unique]),
             ]

REQUIRED_PTS = []

# Copyright 2024 The TEMPO Collaboration
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

import itertools

def run_all(tests, verbose=True):
    results_list_list = []
    for performance_function, parameters_list in tests:
        results_list = []
        if verbose:
            print(f"# Run {performance_function.__name__}:")
        for i, parameters in enumerate(parameters_list):
            param_comb = list(itertools.product(*parameters))
            if verbose:
                print(f"## {i+1}/{len(parameters_list)}: {len(param_comb)} parameter sets:")
            results = []
            for j, params in enumerate(param_comb):
                if verbose:
                    print(f"### parameter set {j+1} of {len(param_comb)}:")
                results.append(performance_function(*params))
            results_list.append(results)
        results_list_list.append(results_list)
    return results_list_list


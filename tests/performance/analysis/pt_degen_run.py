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
Script to run the PT performance analysis.
"""

import sys
sys.path.insert(0,'.')

import dill

from tests.performance.run_all import run_all
from tests.performance.pt_degen import ALL_TESTS

# -- computation --------------------------------------------------------------

all_results = run_all(ALL_TESTS)
with open('./tests/data/performance_results/pt_degen.pkl', 'wb') as f:
    dill.dump(all_results, f)

print("... all done.")

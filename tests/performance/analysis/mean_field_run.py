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
Script to run the mean-field performance analysis.
"""

import sys
sys.path.insert(0,'.')

import dill

from tests.data.generate_pts import generate_spin_boson_pt
from tests.data.generate_pts import spin_boson_pt_exists
from tests.performance.run_all import run_all
from tests.performance.mean_field import ALL_TESTS
from tests.performance.mean_field import REQUIRED_PTS

# -- preparation --------------------------------------------------------------

print("# Prepare computations:")

for pt_name in REQUIRED_PTS:
    if spin_boson_pt_exists(pt_name):
        print(f"Process tensor '{pt_name}' already exists. ")
    else:
        print(f"Generating process tensor '{pt_name}' ... ")
        generate_spin_boson_pt(pt_name)

print("... preparation done.")

# -- computation --------------------------------------------------------------

all_results = run_all(ALL_TESTS)
with open('./tests/data/performance_results/mean-field_results.pkl', 'wb') as f:
    dill.dump(all_results, f)

print("... all done.")

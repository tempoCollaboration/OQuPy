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
"""
Skript to prepare the PT-TEBD performance analysis.
"""

import sys
sys.path.insert(0,'.')

from tests.data.generate_pts import generate_spin_boson_pt
from tests.data.generate_pts import spin_boson_pt_exists
from tests.performance.pt_tebd import REQUIRED_PTS

# -----------------------------------------------------------------------------

for pt_name in REQUIRED_PTS:
    if spin_boson_pt_exists(pt_name):
        print(f"Process tensor '{pt_name}' already exists. ")
    else:
        print(f"Generating process tensor '{pt_name}' ... ")
        generate_spin_boson_pt(pt_name)

print("All done.")

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
Script to run the multi-time correlations performance analysis.
"""

import sys
sys.path.insert(0,'.')

import oqupy
import dill
import os
import numpy as np
import re

from tests.performance.run_all import run_all
from tests.performance.nt_corrs import ALL_TESTS
from tests.performance.nt_corrs import REQUIRED_PTS

PT_DIR_PATH = "./tests/data/process_tensors/"


def pt_3ls_exists(name, pt_dir = PT_DIR_PATH):
      """Check the existence of a precomputed process tensor in the process tensor directory."""
      return os.path.isfile(os.path.join(pt_dir,f"{name}.hdf5"))

def generate_3ls_pt(name, pt_dir = PT_DIR_PATH):
      """
      Generate a process tensor from 'name' using PT-TEMPO and write it into the
      process tensor directory 'pt_dir'.

      """
      P_1 = np.array([[0., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 0.]], dtype=complex)

      P_2 = np.array([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 1.]], dtype=complex)


      res = re.match('3ls_alpha0.1_zeta1.0_T13.09_cutoff3.04exp_tcut50.0_dt0.1_steps80_epsrel([\d]+)', name)
      if res is None:
          raise ValueError(f"Invalid PT name '{name}'")
      p = {
              'epsrelExp':int(res[1]),
      }

      file_path = os.path.join(PT_DIR_PATH,f"{name}.hdf5")

      dt = 0.1
      steps = 80
      epsrel = 10**(-p['epsrelExp'])
      end_time = dt*steps
      dkmax = 500

      correlations = oqupy.PowerLawSD(alpha=0.1,
                                      zeta=1.0,
                                      cutoff=3.04,
                                      cutoff_type='exponential',
                                      temperature=13.09)
      bath = oqupy.Bath(P_1 - P_2, correlations)
      pt_tempo_parameters = oqupy.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

      pt = oqupy.pt_tempo_compute(
            bath=bath,
            start_time=0.0,
            end_time=end_time,
            parameters=pt_tempo_parameters,
            process_tensor_file=file_path,
            overwrite=True,
      )

      pt.close()


# -- preparation --------------------------------------------------------------

print("# Prepare computations:")


for pt_name in REQUIRED_PTS:
    if pt_3ls_exists(pt_name):
        print(f"Process tensor '{pt_name}' already exists. ")
    else:
        print(f"Generating process tensor '{pt_name}' ... ")
        generate_3ls_pt(pt_name)

print("... preparation done.")

# -- computation --------------------------------------------------------------

all_results = run_all(ALL_TESTS)
with open('./tests/data/performance_results/nt_corrs.pkl', 'wb') as f:
    dill.dump(all_results, f)

print("... all done.")

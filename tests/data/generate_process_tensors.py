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
# """
# Skript to generate process tensors for testing purposes.
# """
import sys
sys.path.insert(0,'.')

import os
import re
import numpy as np
import oqupy

# """
# ToDo: Generate process tensors that will then be made available for testing
#       purposes under /tests/data/process_tensors.
#       The generated process tensors should be stored on a public data record,
#       (e.g. Zenodo) such that they can be loaded into
#       /test/data/process_tensors when needed.

#       The process tensors to generate should be combinations of
#       Coupling strength alpha: 0.08 / 0.16 / 0.32
#       Spectral density exponent: 0.5 / 1.0 / 3.0
#       Temperature: 0.0 / 0.8 / 1.6
#       Cutoff: gaussian
#       dt: 2**(-4) / 2**(-5) / 2**(-6)
#       steps: 2**7 / 2**8 / 2**9

#       The filenames should then have a form like this example:
#       "boson_alpha0.16_zeta3.0_T0.8_gaussian_dt05_steps08.hdf5"
# """

PT_DIR_PATH = "./tests/data/process_tensors/"


def generate_spin_boson_pt(name):
      # e.g.: name = "spinBoson_alpha0.08_zeta1.0_T0.8_cutOff1.0expon_tcut2.0_dt04_steps06_epsrel15"
      res = re.match('spinBoson_alpha([\d.]+)_zeta([\d.]+)_T([\d.]+)_cutoff([\d.]+)([a-z]+)_tcut([\d.]+)_dt([\d]+)_steps([\d]+)_epsrel([\d]+)', name)
      p = {
            'alpha':float(res[1]),
            'zeta':float(res[2]),
            'T':float(res[3]),
            'cutoff':float(res[4]),
            'cutoffType':res[5],
            'tcut':float(res[6]),
            'dtExp':int(res[7]),
            'stepsExp':int(res[8]),
            'epsrelExp':int(res[9]),
      }
      # filename = "spinBoson" \
      #       +f"_alpha{p['alpha']:4.2f}_zeta{p['zeta']:3.1f}_T{p['T']:3.1f}_" \
      #       +f"cutOff{p['cutoff']:3.1f}{p['cutoffType']}_tcut{p['tcut']:3.1f}_" \
      #       +f"dt{p['dtExp']:02d}_steps{p['stepsExp']:02d}_epsrel{p['epsrelExp']:02d}" \
      #       +".hdf5"
      file_path = os.path.join(PT_DIR_PATH,f"{name}.hdf5")

      dt = 2**(-p['dtExp'])
      steps = 2**p['stepsExp']
      epsrel = 2**(-p['epsrelExp'])
      end_time = 2**(p['stepsExp']-p['dtExp'])

      if p['cutoffType'] == 'gauss':
            cutoff_type = 'gaussian'
      elif p['cutoffType'] == 'expon':
            cutoff_type = 'exponential'
      else:
            raise ValueError(f"Cutoff Type {p['cutoffType']} not known.")



      correlations = oqupy.PowerLawSD(alpha=p['alpha'],
                                      zeta=p['zeta'],
                                      cutoff=p['cutoff'],
                                      cutoff_type=cutoff_type,
                                      temperature=p['T'])
      bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)
      pt_tempo_parameters = oqupy.TempoParameters(dt=dt, tcut=p['tcut'], epsrel=epsrel)

      pt = oqupy.pt_tempo_compute(
            bath=bath,
            start_time=0.0,
            end_time=end_time,
            parameters=pt_tempo_parameters,
            process_tensor_file=file_path,
            overwrite=True,
      )

      pt.close()


name = "spinBoson_alpha0.32_zeta1.0_T0.0_cutoff1.0expon_tcut4.0_dt04_steps07_epsrel17"

generate_spin_boson_pt(name)
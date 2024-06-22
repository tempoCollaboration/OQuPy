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
Module to generate process tensors for testing purposes.
"""
import sys
sys.path.insert(0,'.')

import os
import re
import numpy as np
import oqupy

PT_DIR_PATH = "./tests/data/process_tensors/"


# -----------------------------------------------------------------------------

def spin_boson_pt_exists(name, pt_dir = PT_DIR_PATH):
      """Check the existence of a precomputed process tensor in the process tensor directory."""
      return os.path.isfile(os.path.join(pt_dir,f"{name}.hdf5"))

def parameters_from_name(name):
      res = re.match('spinBoson_alpha([\d.]+)_zeta([\d.]+)_T([\d.]+)_cutoff([\d.]+)([a-z]+)_tcut([\d.]+)_dt([\d]+)_steps([\d]+)_epsrel([\d]+)_*(.*)', name)
      if res is None:
          raise ValueError(f"Invalid PT name '{name}'")
      coupling = res[10] if res[10]!= '' else 'z'  
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
            'coupling':coupling,
      }
      return p

def bath_from_parameters(p):
      cutoff_map = {'gss': 'gaussian', 'exp': 'exponential', 'hrd':'hard'}

      try:
          cutoff_type = cutoff_map[p['cutoffType']]
      except KeyError:
          raise ValueError(f"Cutoff Type {p['cutoffType']} not known.")

      correlations = oqupy.PowerLawSD(alpha=p['alpha'],
                                      zeta=p['zeta'],
                                      cutoff=p['cutoff'],
                                      cutoff_type=cutoff_type,
                                      temperature=p['T'])
      bath = oqupy.Bath(oqupy.operators.sigma(p['coupling'])/2.0, correlations)
      return bath

def generate_spin_boson_pt(name, pt_dir = PT_DIR_PATH):
      """
      Generate a spin boson process tensor from 'name' using PT-TEBD and write it into the
      process tensor directory 'pt_dir'.

      The name 'name' has to be of the form:
      'spinBoson_alpha$.$$_zeta$.$_T$.$_cutoff$.$%%%_tcut$.$_dt$$_steps$$_epsrel$$'
                      A---     B--  C--       D--E--     F--   G-      H-       I-
      where '$' are digits and '%' are lowercase letters.

      For example:
      'spinBoson_alpha0.08_zeta1.0_T0.8_cutoff1.0exp_tcut2.0_dt04_steps06_epsrel15'
                      A---     B--  C--       D--E--     F--   G-      H-       I-
      is then, for example, translated into the following parameters:
      A ... alpha        = 0.08
      B ... zeta         = 1.0
      C ... temperature  = 0.8
      D ... cutoff       = 1.0
      E ... cutoffType   = 'exponential' # (exp-> exponential, gss-> gaussian, hrd-> hard)
      F ... tcut         = 2.0
      G ... dt           = 2**(-4)
      H ... steps        = 2**(+6)
      I ... epsrel       = 2**(-15)

      """
      p = parameters_from_name(name)
      bath = bath_from_parameters(p)

      file_path = os.path.join(PT_DIR_PATH,f"{name}.hdf5")

      dt = 2**(-p['dtExp'])
      steps = 2**p['stepsExp']
      epsrel = 2**(-p['epsrelExp'])
      end_time = 2**(p['stepsExp']-p['dtExp'])

      pt_tempo_parameters = oqupy.TempoParameters(dt=dt, tcut=p['tcut'], epsrel=epsrel)

      pt = oqupy.pt_tempo_compute(
            bath=bath,
            start_time=0.0,
            end_time=end_time,
            parameters=pt_tempo_parameters,
            process_tensor_file=file_path,
            overwrite=True,
            description=f"automatically generated from name: '{name}'."
      )
      pt.name = "spin boson model"

      pt.close()
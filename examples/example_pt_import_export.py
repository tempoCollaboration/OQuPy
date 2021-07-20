"""
NOTE: This file should be executed from the repositories main directory.
"""
import sys
sys.path.insert(0,'.')
import os
from time import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import time_evolving_mpo as tempo
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

# -- Preparations to Demonstrate Process Tensor In/Export ---------------------

zeta = 3
Omega = 1.0
omega_cutoff = 5.0
alpha = 0.1
temperature = 0.0
start_time = 0.0
end_time = 2.0
initial_state = tempo.operators.spin_dm("y+")

system = tempo.System(0.0 * Omega * tempo.operators.sigma("x"))

correlations = tempo.PowerLawSD(alpha=alpha,
                                zeta=zeta,
                                cutoff=omega_cutoff,
                                cutoff_type='exponential',
                                temperature=temperature,
                                max_correlation_time=2.0)
bath = tempo.Bath(0.5 * tempo.operators.sigma("z"), correlations)

dt = 0.1
dkmax = 3
epsrel = 1.0e-6
tempo_parameters = tempo.TempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)
pt_tempo_parameters = tempo.PtTempoParameters(dt=dt, dkmax=dkmax, epsrel=epsrel)

# -- In/Export of Process Tensors  --------------------------------------------

pt_filename_A = "./examples/data/pt_temp_A.hdf5"
pt_filename_B = "./examples/data/pt_temp_B.hdf5"

# the pt_tempo computation will return a SimpleProceeTensor by default
simple_pt = tempo.pt_tempo_compute(
    bath=bath,
    start_time=start_time,
    end_time=end_time,
    parameters=pt_tempo_parameters,
    progress_type='silent')

# this will write the SimpleProcessTensor to a file
simple_pt.export(filename=pt_filename_A, overwrite=True)

# this reads the file back into a new SimpleProcessTensor
simple_A_pt = \
    tempo.import_process_tensor(pt_filename_A, process_tensor_type="simple")

# SimpleProcessTensor objects store the entire process tensor in memory.
# Therefore it is advisable to delete it manually when no longer needed.
del simple_pt
del simple_A_pt

# this reads the file back into a new FileProcessTensor
#    (the FileProcessTensor object only loads the requested parts of the
#     process tensor and thus saves memory)
file_A_pt = \
    tempo.import_process_tensor(pt_filename_A, process_tensor_type="file")

# after using this FileProcessTensor we should (idealy) close it:
file_A_pt.close()

# changing or removing the file is not possible because it is opened in
# read only mode

try:
    dim = file_A_pt.get_bond_dimensions()[1]
    file_A_pt.set_lam_tensor(0, np.identity(dim))
except:
    print("Told you! (can't edit)")

try:
    file_A_pt.remove()
except:
    print("Told you! (can't remove)")

# we can tell the pt_tempo computation to return a FileProcessTensor directly
file_B_pt = tempo.pt_tempo_compute(
    bath=bath,
    start_time=start_time,
    end_time=end_time,
    parameters=pt_tempo_parameters,
    process_tensor_file=pt_filename_B,
    progress_type='silent')

# because we've created this file we can also change it's data
dim = file_B_pt.get_bond_dimensions()[1]
file_B_pt.set_lam_tensor(0, np.identity(dim))
file_B_pt.close()

# we can overwrite file B with a new computation
file_B_pt = tempo.pt_tempo_compute(
    bath=bath,
    start_time=start_time,
    end_time=end_time,
    parameters=pt_tempo_parameters,
    process_tensor_file=pt_filename_B,
    overwrite=True,
    progress_type='silent')

# we can also delete a file on disk that we've created
file_B_pt.remove()

# we can also create a temporary file (in the systems standard tmp-directory):
file_temp_pt = tempo.pt_tempo_compute(
    bath=bath,
    start_time=start_time,
    end_time=end_time,
    parameters=pt_tempo_parameters,
    process_tensor_file=True,
    progress_type='silent')

# we can find out what the filename of the temporary file is
print(f"TEMP filename = {file_temp_pt.filename}")

# the temp file is NOT automatically removed, so this should be done manually
# if desired
file_temp_pt.remove()

# -----------------------------------------------------------------------------

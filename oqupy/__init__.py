"""
A Python package to efficiently simulate non-Markovian open quantum systems
with process tensors.

This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly strongly
coupled to structured environments. It facilitates the convenient application
of several numerical methods that combine the conceptional advantages of the
process tensor framework [1], with the numerical efficiency of tensor networks.

OQuPy includes numerically exact methods (i.e. employing only numerically well
controlled approximations) for the non-Markovian dynamics and multi-time
correlations of ...
- quantum systems coupled to a single environment [2-4],
- quantum systems coupled to multiple environments [5],
- interacting chains of non-Markovian open quantum systems [6], and
- ensembles of open many-body systems with many-to-one coupling [7].

Furthermore, OQuPy implements methods to ...
- optimize control protocols for non-Markovian open quantum systems [8,9],
- compute the dynamics of an non-Markovian environment [10], and
- obtain the thermal state of a strongly couled quantum system [11].

[1]  Pollock et al., [Phys. Rev. A 97, 012127]
     (https://doi.org/10.1103/PhysRevA.97.012127) (2018).
[2]  Strathearn et al., [New J. Phys. 19(9), p.093009]
     (https://doi.org/10.1088/1367-2630/aa8744) (2017).
[3]  Strathearn et al., [Nat. Commun. 9, 3322]
     (https://doi.org/10.1038/s41467-018-05617-3) (2018).
[4]  Jørgensen and Pollock, [Phys. Rev. Lett. 123, 240602]
     (https://doi.org/10.1103/PhysRevLett.123.240602) (2019).
[5]  Gribben et al., [PRX Quantum 3, 10321]
     (https://doi.org/10.1103/PRXQuantum.3.010321) (2022).
[6]  Fux et al., [Phys. Rev. Research 5, 033078 ]
     (https://doi.org/10.1103/PhysRevResearch.5.033078}) (2023).
[7]  Fowler-Wright et al., [Phys. Rev. Lett. 129, 173001]
     (https://doi.org/10.1103/PhysRevLett.129.173001) (2022).
[8]  Fux et al., [Phys. Rev. Lett. 126, 200401]
     (https://doi.org/10.1103/PhysRevLett.126.200401) (2021).
[9]  Butler et al., [Phys. Rev. Lett. 132, 060401 ]
     (https://doi.org/10.1103/PhysRevLett.132.060401}) (2024).
[10] Gribben et al., [Quantum, 6, 847]
     (https://doi.org/10.22331/q-2022-10-25-847) (2022).
[11] Chiu et al., [Phys. Rev. A 106, 012204]
     (https://doi.org/10.1103/PhysRevA.106.012204}) (2022).

"""
from oqupy.version import __version__

# all API functionallity is in __all__
__all__ = [
    'AugmentedMPS',
    'Bath',
    'ChainControl',
    'CustomCorrelations',
    'CustomSD',
    'compute_correlations',
    'compute_correlations_nt',
    'compute_dynamics',
    'compute_dynamics_with_field',
    'compute_gradient_and_dynamics',
    'Control',
    'Dynamics',
    'FileProcessTensor',
    'GibbsParameters',
    'GibbsTempo',
    'gibbs_tempo_compute',
    'guess_tempo_parameters',
    'helpers',
    'import_process_tensor',
    'MeanFieldDynamics',
    'MeanFieldSystem',
    'MeanFieldTempo',
    'operators',
    'ParameterizedSystem',
    'PowerLawSD',
    'PtTebd',
    'PtTebdParameters',
    'PtTempo',
    'pt_tempo_compute',
    'SimpleProcessTensor',
    'state_gradient',
    'System',
    'SystemChain',
    'Tempo',
    'tempo_compute',
    'TempoParameters',
    'TimeDependentSystem',
    'TimeDependentSystemWithField',
    'TrivialProcessTensor',
    'TwoTimeBathCorrelations',
    ]

# -- Modules in alphabetical order --------------------------------------------

from oqupy.bath import Bath

from oqupy.bath_dynamics import TwoTimeBathCorrelations

from oqupy.system_dynamics import compute_correlations
from oqupy.system_dynamics import compute_correlations_nt
from oqupy.system_dynamics import compute_dynamics
from oqupy.system_dynamics import compute_dynamics_with_field

from oqupy.control import Control
from oqupy.control import ChainControl

from oqupy.bath_correlations import CustomCorrelations
from oqupy.bath_correlations import CustomSD
from oqupy.bath_correlations import PowerLawSD

from oqupy.dynamics import Dynamics
from oqupy.dynamics import MeanFieldDynamics

from oqupy.gradient import state_gradient
from oqupy.gradient import compute_gradient_and_dynamics

from oqupy import helpers

from oqupy.mps_mpo import AugmentedMPS

from oqupy import operators

from oqupy.process_tensor import import_process_tensor
from oqupy.process_tensor import TrivialProcessTensor
from oqupy.process_tensor import SimpleProcessTensor
from oqupy.process_tensor import FileProcessTensor

from oqupy.pt_tebd import PtTebd
from oqupy.pt_tebd import PtTebdParameters

from oqupy.system import System
from oqupy.system import SystemChain
from oqupy.system import TimeDependentSystem
from oqupy.system import TimeDependentSystemWithField
from oqupy.system import MeanFieldSystem
from oqupy.system import ParameterizedSystem

from oqupy.pt_tempo import PtTempo
from oqupy.pt_tempo import pt_tempo_compute

from oqupy.tempo import Tempo
from oqupy.tempo import TempoParameters
from oqupy.tempo import GibbsTempo
from oqupy.tempo import GibbsParameters
from oqupy.tempo import MeanFieldTempo
from oqupy.tempo import guess_tempo_parameters
from oqupy.tempo import tempo_compute
from oqupy.tempo import gibbs_tempo_compute

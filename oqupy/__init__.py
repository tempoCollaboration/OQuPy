"""
A Python 3 package to efficiently compute non-Markovian open quantum systems.

This open source project aims to facilitate versatile numerical tools to
efficiently compute the dynamics of quantum systems that are possibly strongly
coupled to a structured environment. It allows to conveniently apply the so
called time evolving matrix product operator method (TEMPO) [1], as well as
the process tensor TEMPO method (PT-TEMPO) [2].

[1] A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
    B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
    time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
[2] G. E. Fux, E. Butler, P. R. Eastham, B. W. Lovett, and
    J. Keeling, *Efficient exploration of Hamiltonian parameter space for
    optimal control of non-Markovian open quantum systems*, arXiv2101.?????
    (2021).
"""
from oqupy.version import __version__

# all API functionallity is in __all__
__all__ = [
    'Bath',
    'CustomCorrelations',
    'CustomSD',
    'Dynamics',
    'file_formats',
    'FileProcessTensor',
    'guess_pt_tempo_parameters',
    'guess_tempo_parameters',
    'helpers',
    'import_dynamics',
    'import_process_tensor',
    'NumericsError',
    'NumericsWarning',
    'operators',
    'PowerLawSD',
    'PtTempo',
    'pt_tempo_compute',
    'SimpleProcessTensor',
    'System',
    'Tempo',
    'tempo_compute',
    'TempoParameters',
    'TimeDependentSystem',
    'TrivialProcessTensor',
    ]

# -- Modules in alphabetical order --------------------------------------------

from oqupy.bath import Bath

# from oqupy.control import Control

from oqupy.dynamics import Dynamics
from oqupy.dynamics import import_dynamics

from oqupy.exceptions import NumericsError
from oqupy.exceptions import NumericsWarning

import oqupy.file_formats

import oqupy.operators

import oqupy.helpers

from oqupy.process_tensor import import_process_tensor
from oqupy.process_tensor import TrivialProcessTensor
from oqupy.process_tensor import SimpleProcessTensor
from oqupy.process_tensor import FileProcessTensor

from oqupy.tempo.pt_tempo import PtTempo
from oqupy.tempo.pt_tempo import PtTempoParameters
from oqupy.tempo.pt_tempo import guess_pt_tempo_parameters
from oqupy.tempo.pt_tempo import pt_tempo_compute

from oqupy.correlations import CustomCorrelations
from oqupy.correlations import CustomSD
from oqupy.correlations import PowerLawSD

from oqupy.system import System
from oqupy.system import TimeDependentSystem

from oqupy.tempo.tempo import Tempo
from oqupy.tempo.tempo import TempoParameters
from oqupy.tempo.tempo import guess_tempo_parameters
from oqupy.tempo.tempo import tempo_compute

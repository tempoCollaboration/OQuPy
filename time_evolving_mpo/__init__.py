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
from time_evolving_mpo.version import __version__

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

from time_evolving_mpo.bath import Bath

# from time_evolving_mpo.control import Control

from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.dynamics import import_dynamics

from time_evolving_mpo.exceptions import NumericsError
from time_evolving_mpo.exceptions import NumericsWarning

import time_evolving_mpo.file_formats

import time_evolving_mpo.operators

import time_evolving_mpo.helpers

from time_evolving_mpo.process_tensor import import_process_tensor
from time_evolving_mpo.process_tensor import TrivialProcessTensor
from time_evolving_mpo.process_tensor import SimpleProcessTensor
from time_evolving_mpo.process_tensor import FileProcessTensor

from time_evolving_mpo.pt_tempo import PtTempo
from time_evolving_mpo.pt_tempo import PtTempoParameters
from time_evolving_mpo.pt_tempo import guess_pt_tempo_parameters
from time_evolving_mpo.pt_tempo import pt_tempo_compute

from time_evolving_mpo.correlations import CustomCorrelations
from time_evolving_mpo.correlations import CustomSD
from time_evolving_mpo.correlations import PowerLawSD

from time_evolving_mpo.system import System
from time_evolving_mpo.system import TimeDependentSystem

from time_evolving_mpo.tempo import Tempo
from time_evolving_mpo.tempo import TempoParameters
from time_evolving_mpo.tempo import guess_tempo_parameters
from time_evolving_mpo.tempo import tempo_compute

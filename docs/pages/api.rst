API Outline
===========

The TEMPO Collaboration is continuously developing extensions to the original
TEMPO algorithm to make this package applicable to a wider set of scenarios
(the process tensor approach is one such extension). This calls for a flexible
API design to allow to reuse the same objects with different algorithms. We
therefore choose an almost fully object oriented approach and separate the
code into three layers:

1. **The physical layer**: Consists of objects that describe physical
   quantities, like a system (with a specific Hamiltonian) or the spectral
   density of an environment.
2. **The algorithms layer**: Gathers the information from the physical
   layer and feeds it (with particular simulation parameters) into the backend.
   It has the opportunity to make use of specific, extra information from the
   physical layer.
3. **The backend layer**: Is the part of the code where the task has been
   reduced to a mathematically well defined computation, such as a specific
   tensor network or an integral. This not only allows to compare the
   performance of different implementations, it also allows to write hardware
   specialised implementations (single CPU, cluster, GPU) while keeping the
   other two layers untouched.

Also, we try to supply some handy utilities such as shortcuts for the
Pauli operators for example.


The physical layer
------------------

class :class:`time_evolving_mpo.system.BaseSystem`
  Abstract class representing a quantum system of interest.

  class :class:`time_evolving_mpo.system.System`
    Encodes system Hamiltonian and possibly some additional Markovian decay.

  class :class:`time_evolving_mpo.system.TimeDependentSystem`
    Encodes a time dependent system Hamiltonian and possibly some additional
    time dependent Markovian decay.

class :class:`time_evolving_mpo.correlations.BaseCorrelations`
  Abstract class representing the environments auto-correlations.

  class :class:`time_evolving_mpo.correlations.CustomCorrelations`
    Encode an explicitly given environment auto-correlation function.

  class :class:`time_evolving_mpo.correlations.CustomSD`
    Encodes the auto-correlations for a given spectral density.

  class :class:`time_evolving_mpo.correlations.PowerLawSD`
    Encodes the auto-correlations for a given spectral density of a power law
    form.

class :class:`time_evolving_mpo.bath.Bath`
  Bundles a :class:`time_evolving_mpo.correlations.BaseCorrelations` object
  together with a coupling operator.

The algorithms layer
--------------------

TEMPO
*****

class :class:`time_evolving_mpo.tempo.TempoParameters`
  Stores a set of parameters for a TEMPO computation.

class :class:`time_evolving_mpo.tempo.Tempo`
  Class to facilitate a TEMPO computation.

  method :meth:`time_evolving_mpo.tempo.Tempo.compute`
    Method that carries out a TEMPO computation.

class :class:`time_evolving_mpo.dynamics.Dynamics`
  Object that encodes the time evolution of a system (with discrete time steps).

function :func:`time_evolving_mpo.tempo.guess_tempo_parameters`
  Function that chooses an appropriate set of parameters for a particular
  TEMPO computation.


PT-TEMPO
********

class :class:`time_evolving_mpo.pt_tempo.PtTempoParameters`
  Stores a set of parameters for a PT-TEMPO computation.

class :class:`time_evolving_mpo.pt_tempo.PtTempo`
  Class to facilitate a PT-TEMPO computation.

  method :meth:`time_evolving_mpo.pt_tempo.PtTempo.compute`
    Method that carries out a PT-TEMPO computation.

class :class:`time_evolving_mpo.process_tensor.ProcessTensor`
  Object that encodes a so called process tensor (which captures all possible
  Markovian and non-Markovian interactions between some system and an
  environment).


The backend layer
-----------------

Currently the only backend available is the ``'tensor-network'`` backend,
makes use of the external python package
`TensorNetwork <https://github.com/google/TensorNetwork>`_ to carry out the
heavy lifting of the tensor network computations. This package itself can,
however, be configured to use different tensor network backends
(such as "numpy", "tensorflow" and "pytorch"). All the classes belonging to the
algorithm layer allow you to choose the backend and its configuration
(with the parameters ``backend`` and ``backend_config``).

The default uses:

.. code-block:: python3

  backend = 'tensor-network'
  backend_config = {'backend':'numpy'}


Utillities
----------

module :mod:`time_evolving_mpo.operators`
  Supplies several commonly used operators, such as the Pauli matrices and spin
  density matrices.

function :func:`time_evolving_mpo.helpers.plot_correlations_with_parameters`
  A helper function to plot an auto-correlation function and the sampling
  points given by a set of parameters for a TEMPO computation.

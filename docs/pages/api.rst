API Outline
===========

The TEMPO Collaboration is continuously developing new methods to make this
package applicable to a wider set of scenarios. This calls for a flexible
API design to allow to reuse the same objects with different algorithms. We
therefore choose an almost fully object oriented approach. The functions and
objects fall into 4 categories:

1. **Physical**: Consists of objects that describe physical quantities, like
   for example a system Hamiltonian or the spectral density of an environment.
2. **Methods**: Gathers the information from physical objects and applies a
   numerical method using particular simulation parameters.
3. **Results**: Encode the results of a computation. Unlike physical objects,
   these objects may depend on computational parameters (like for example a
   specific time step length).
4. **Utilities**: Supplies some handy utilities such as shorthands for the
   Pauli operators.

Physical
--------

Systems
*******

class :class:`oqupy.system.BaseSystem`
  Abstract class representing a quantum system of interest.

  class :class:`oqupy.system.System`
    Encodes system Hamiltonian and possibly some additional Markovian decay.

  class :class:`oqupy.system.TimeDependentSystem`
    Encodes a time dependent system Hamiltonian and possibly some additional
    time dependent Markovian decay.

  class :class:`oqupy.system.TimeDependentSystemWithField`
    Encodes a system Hamiltonian (and possibly some additional time dependent
    Markovian decay) that depends on both time and the expectation value of
    a field (a complex scalar) to which the system couples.

class :class:`oqupy.system.MeanFieldSystem`
  Encodes a collection of time dependent systems that couple to a common
  field which evolves according to a prescribed equation of motion.

class :class:`oqupy.system.SystemChain`
  Encodes a 1D chain of systems and possibly some additional Markovian decay.


Control
*******

class :class:`oqupy.control.Control`
  Encodes control operations on `oqupy.system.BaseSystem` objects.

class :class:`oqupy.control.ChainControl`
  Encodes control operations on `oqupy.system.SystemChain` objects.


Environment
***********

class :class:`oqupy.correlations.BaseCorrelations`
  Abstract class representing the environments auto-correlations.

  class :class:`oqupy.correlations.CustomCorrelations`
    Encode an explicitly given environment auto-correlation function.

  class :class:`oqupy.correlations.CustomSD`
    Encodes the auto-correlations for a given spectral density.

  class :class:`oqupy.correlations.PowerLawSD`
    Encodes the auto-correlations for a given spectral density of a power law
    form.

class :class:`oqupy.bath.Bath`
  Bundles a :class:`oqupy.correlations.BaseCorrelations` object
  together with a coupling operator.


Methods
-------

TEMPO
*****
(Time Evolving Matrix Product Operator)

class :class:`oqupy.tempo.TempoParameters`
  Stores a set of parameters for a TEMPO computation.

class :class:`oqupy.tempo.Tempo`
  Class to facilitate a TEMPO computation.

  method :meth:`oqupy.tempo.Tempo.compute`
    Method that carries out a TEMPO computation and creates a
    :class:`oqupy.dynamics.Dynamics` object.

class :class:`oqupy.tempo.MeanFieldTempo`
    Class to facilitate a TEMPO computation with concurrent evolution of
    a coherent field.

    method :meth:`oqupy.tempo.MeanFieldTempo.compute`
      Method that carries out a TEMPO computation while evolving a coherent
      field, and creates a :class:`oqupy.dynamics.MeanFieldDyanmics` object.

function :func:`oqupy.tempo.guess_tempo_parameters`
  Function that chooses an appropriate set of parameters for a particular
  TEMPO computation.


PT-TEMPO
********
(Process Tensor - Time Evolving Matrix Product Operator)

class :class:`oqupy.pt_tempo.PtTempo`
  Class to facilitate a PT-TEMPO computation.

  method :meth:`oqupy.pt_tempo.PtTempo.compute`
    Method that carries out a PT-TEMPO computation and creates an
    :class:`oqupy.process_tensor.BaseProcessTensor` object.


Process Tensor Applications
***************************

function :func:`oqupy.contractions.compute_dynamics`
  Compute a :class:`oqupy.dynamics.Dynamics` object for given
  :class:`oqupy.system.System` or
  :class:`oqupy.system.TimeDependentSystem` and
  :class:`oqupy.control.Control` and
  :class:`oqupy.process_tensor.BaseProcessTensor` objects.

function :func:`oqupy.contractions.compute_dynamics_with_field`
  Compute a :class:`oqupy.dynamics.MeanFieldDynamics` object for given
  :class:`oqupy.system.MeanFieldSystem` and list of
  :class:`oqupy.control.Control` objects and list of
  :class:`oqupy.process_tensor.BaseProcessTensor` objects.
  
function :func:`oqupy.contractions.compute_correlations_nt`
  Compute ordered multi-time correlations for given
  :class:`oqupy.system.BaseSystem` and
  :class:`oqupy.process_tensor.BaseProcessTensor` objects.

function :func:`oqupy.contractions.compute_correlations`
  Compute two time correlations for given
  :class:`oqupy.system.BaseSystem` and
  :class:`oqupy.process_tensor.BaseProcessTensor` objects.

class :class:`oqupy.bath_dynamics.TwoTimeBathCorrelations`
  Class to facilitate calculation of two-time bath correlations.

  method :meth:`oqupy.bath_dynamics.TwoTimeBathCorrelations.occupation`
    Function to calculate the change in bath occupation in a particular
    bandwidth.

  method :meth:`oqupy.bath_dynamics.TwoTimeBathCorrelations.correlation`
    Function to calculate two-time correlation function between two
    frequency bands of a bath.


PT-TEBD
*******
(Process Tensor - Time Evolving Block Decimation)

class :class:`oqupy.pt_tebd.PtTebdParameters`
  Stores a set of parameters for a PT-TEBD computation.

class :class:`oqupy.pt_tebd.PtTebd`
  Class to facilitate a PT-TEBD computation.

  method :meth:`oqupy.pt_tebd.PtTebd.compute`
    Method that carries out a PT-TEMPO computation and returns an results
    dictionary.



Results
-------

class :class:`oqupy.dynamics.Dynamics`
  Object that encodes the discretized evolution of the reduced density matrix
  of a system.

class :class:`oqupy.dynamics.MeanFieldDynamics`
  Object that encodes the discretized evolution of the reduced density matrix
  of one or more time-dependent systems together with that of a classical field
  coupled to the systems.

class :class:`oqupy.process_tensor.BaseProcessTensor`
  Object that encodes a so called process tensor (which captures all possible
  Markovian and non-Markovian interactions between some system and an
  environment).


Utilities
---------

module :mod:`oqupy.operators`
  Supplies several commonly used operators, such as the Pauli matrices and spin
  density matrices.

function :func:`oqupy.helpers.plot_correlations_with_parameters`
  A helper function to plot an auto-correlation function and the sampling
  points given by a set of parameters for a TEMPO computation.

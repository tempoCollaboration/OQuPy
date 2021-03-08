# Copyright 2021 The TEMPO Collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file  in compliance with the License.
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
Module on for the original time evolving matrix product operator (TEMPO)
algorithm. This code is based on [Strathearn2018].

**[Strathearn2018]**
A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
"""

import sys
from typing import Callable, Dict, Optional, Text
import warnings
from copy import copy

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

from time_evolving_mpo.backends.backend_factory import get_tempo_backend
from time_evolving_mpo.bath import Bath
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NpDtype, MAX_DKMAX, DEFAULT_TOLLERANCE
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.util import commutator, acommutator
from time_evolving_mpo.util import get_progress


class TempoParameters(BaseAPIClass):
    r"""
    Parameters for the TEMPO computation.

    Parameters
    ----------
    dt: float
        Length of a time step :math:`\delta t`. - It should be small enough
        such that a trotterisation between the system Hamiltonian and the
        environment it valid, and the environment auto-correlation function
        is reasonably well sampled.
    dkmax: int
        Number of time steps :math:`K\in\mathbb{N}` that should be included in
        the non-Markovian memory. - It must be large
        enough such that :math:`\delta t \times K` is larger than the
        neccessary memory time :math:`\tau_\mathrm{cut}`.
    epsrel: float
        The maximal relative error in the singular value truncation (done
        in the underlying tensor network algorithm). - It must be small enough
        such that the numerical compression (using tensor network algorithms)
        does not truncate relevant correlations.
    """
    def __init__(
            self,
            dt: float,
            dkmax: int,
            epsrel: float,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a TempoParameters object."""
        self.dt = dt
        self.dkmax = dkmax
        self.epsrel = epsrel
        super().__init__(name, description, description_dict)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dt            = {} \n".format(self.dt))
        ret.append("  dkmax         = {} \n".format(self.dkmax))
        ret.append("  epsrel        = {} \n".format(self.epsrel))
        return "".join(ret)

    @property
    def dt(self) -> float:
        """Length of a time step."""
        return self._dt

    @dt.setter
    def dt(self, new_dt: float) -> None:
        try:
            __dt = float(new_dt)
        except Exception as e:
            raise AssertionError("Argument 'dt' must be float.") from e
        assert __dt > 0.0, \
            "Argument 'dt' must be bigger than 0."
        self._dt = __dt

    @property
    def dkmax(self) -> float:
        """Number of time steps that should be included in the non-Markovian
        memory. """
        return self._dkmax

    @dkmax.setter
    def dkmax(self, new_dkmax: float) -> None:
        try:
            if new_dkmax is None:
                __dkmax = None
            else:
                __dkmax = int(new_dkmax)
        except Exception as e:
            raise AssertionError("Argument 'dkmax' must be int or None.") \
                from e
        assert __dkmax is None or __dkmax > 0, \
            "Argument 'dkmax' must be bigger than or equal to 0 or None."
        self._dkmax = __dkmax

    @dkmax.deleter
    def dkmax(self) -> None:
        self._dkmax = None

    @property
    def epsrel(self) -> float:
        """The maximal relative error in the singular value truncation."""
        return self._epsrel

    @epsrel.setter
    def epsrel(self, new_epsrel: float) -> None:
        try:
            __epsrel = float(new_epsrel)
        except Exception as e:
            raise AssertionError("Argument 'epsrel' must be float.") from e
        assert __epsrel > 0.0, \
            "Argument 'epsrel' must be bigger than 0."
        self._epsrel = __epsrel


class Tempo(BaseAPIClass):
    """
    Class representing the entire TEMPO tensornetwork as introduced in
    [Strathearn2018].

    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The Bath (includes the coupling operator to the sytem).
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    initial_state: ndarray
        The initial density matrix of the sytem.
    start_time: float
        The start time.
    backend: str (default = None)
        The name of the backend to use for the computation. If
        `backend` is ``None`` then the default backend is used.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    description_dict: dict (default = None)
        An optional dictionary with descriptive data.
    """
    def __init__(
            self,
            system: BaseSystem,
            bath: Bath,
            parameters: TempoParameters,
            initial_state: ndarray,
            start_time: float,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a Tempo object. """
        self._backend_class, self._backend_config = \
            get_tempo_backend(backend, backend_config)

        assert isinstance(system, BaseSystem), \
            "Argument 'system' must be an instance of BaseSystem."
        self._system = system

        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath

        self._correlations = self._bath.correlations

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        try:
            __initial_state = np.array(initial_state, dtype=NpDtype)
            __initial_state.setflags(write=False)
        except Exception as e:
            raise AssertionError("Initial state must be numpy array.") from e
        assert len(__initial_state.shape) == 2, \
            "Initial state is not a matrix."
        assert __initial_state.shape[0] == \
            __initial_state.shape[1], \
            "Initial state is not a square matrix."
        self._initial_state = __initial_state
        self._dimension = self._initial_state.shape[0]

        try:
            __start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = __start_time

        assert self._bath.dimension == self._dimension and \
            self._system.dimension == self._dimension, \
            "Hilbertspace dimensions are unequal: " \
            + "system ({}), ".format(self._system.dimension) \
            + "initial state ({}), ".format(self._dimension) \
            + "and bath coupling ({}), ".format(self._bath.dimension)

        super().__init__(name, description, description_dict)

        __coupling_comm = commutator(self._bath._coupling_operator)
        __coupling_acomm = acommutator(self._bath._coupling_operator)
        self._coupling_comm = __coupling_comm.diagonal()
        self._coupling_acomm = __coupling_acomm.diagonal()

        self._dynamics = None
        self._backend_instance = None

        self._init_tempo_backend()

    def _init_tempo_backend(self):
        """Create and initialize the tempo backend. """
        dim = self._dimension
        initial_state = self._initial_state.reshape(dim**2)
        influence = self._influence
        unitary_transform = self._bath.unitary_transform
        propagators = self._propagators
        sum_north = np.array([1.0]*(dim**2))
        sum_west = np.array([1.0]*(dim**2))
        dkmax = self._parameters.dkmax
        epsrel = self._parameters.epsrel
        self._backend_instance = self._backend_class(
                initial_state,
                influence,
                unitary_transform,
                propagators,
                sum_north,
                sum_west,
                dkmax,
                epsrel,
                config=self._backend_config)

    def _init_dynamics(self):
        """Create a Dynamics object with metadata from the Tempo object. """
        name = None
        description = "computed from '{}' tempo".format(self.name)
        description_dict = {
            "tempo_type":str(type(self)),
            "tempo_name":self.name,
            "tempo_description":self.description,
            "tempo_description_dict":self.description_dict,
            "parameters_type":str(type(self._parameters)),
            "parameters_name":self._parameters.name,
            "parameters_description":self._parameters.description,
            "parameters_description_dict":self._parameters.description_dict,
            "system_type":str(type(self._system)),
            "system_name":self._system.name,
            "system_description":self._system.description,
            "system_description_dict":self._system.description_dict,
            "bath_type":str(type(self._bath)),
            "bath_name":self._bath.name,
            "bath_description":self._bath.description,
            "bath_description_dict":self._bath.description_dict,
            "correlations_type":str(type(self._correlations)),
            "correlations_name": \
                self._correlations.name,
            "correlations_description": \
                self._correlations.description,
            "correlations_description_dict": \
                self._correlations.description_dict,
            "backend_class":str(self._backend_class),
            "initial_state":self._initial_state,
            "dt":self._parameters.dt,
            "dkmax":self._parameters.dkmax,
            "epsrel":self._parameters.epsrel,
            }
        self._dynamics = Dynamics(name=name,
                                  description=description,
                                  description_dict=description_dict)

    def _influence(self, dk: int):
        """Create the influence functional matrix for a time step distance
        of dk. """
        dt = self._parameters.dt
        dkmax = self._parameters.dkmax

        if dk == 0:
            time_1 = 0.0
            time_2 = None
            shape = "upper-triangle"
        elif dk < 0:
            time_1 = float(dkmax) * dt
            if self._correlations.max_correlation_time is not None:
                time_2 = np.min([
                    float(dkmax-dk) * dt,
                    self._correlations.max_correlation_time])
            else:
                time_2 = float(dkmax-dk) * dt
            shape = "rectangle"
        else:
            time_1 = float(dk) * dt
            time_2 = None
            shape = "square"

        eta_dk = self._correlations.correlation_2d_integral( \
            delta=dt,
            time_1=time_1,
            time_2=time_2,
            shape=shape,
            epsrel=self._parameters.epsrel)
        op_p = self._coupling_acomm
        op_m = self._coupling_comm

        if dk == 0:
            infl = np.diag(np.exp(-op_m*(eta_dk.real*op_m \
                                          + 1j*eta_dk.imag*op_p)))
        else:
            infl = np.exp(-np.outer(eta_dk.real*op_m \
                                  + 1j*eta_dk.imag*op_p, op_m))

        return infl

    def _propagators(self, step: int):
        """Create the system propagators (first and second half) for the time
        step `step`. """
        dt = self._parameters.dt
        t = self._time(step)
        first_step = expm(self._system.liouvillian(t+dt/4.0)*dt/2.0).T
        second_step = expm(self._system.liouvillian(t+dt*3.0/4.0)*dt/2.0).T
        return first_step, second_step

    def _time(self, step: int):
        """Return the time that corresponds to the time step `step`. """
        return self._start_time + float(step)*self._parameters.dt

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension. """
        return copy(self._dimension)

    def compute(
            self,
            end_time: float,
            progress_type: Text = None) -> Dynamics:
        """
        Propagate (or continue to propagete) the TEMPO tensor network to
        time `end_time`.

        Parameters
        ----------
        end_time: float
            The time to which the TEMPO should be computed.
        progress_type: str (default = None)
            The progress report type during the computation. Types are:
            {``'silent'``, ``'simple'``, ``'bar'``}. If `None` then
            the default progress type is used.

        Returns
        -------
        dynamics: Dynamics
            The instance of Dynamics associated with the TEMPO object.
        """
        try:
            __end_time = float(end_time)
        except Exception as e:
            raise AssertionError("End time must be a float.") from e

        dim = self._dimension
        if self._backend_instance.step is None:
            step, state = self._backend_instance.initialize()
            self._init_dynamics()
            self._dynamics.add(self._time(step), state.reshape(dim, dim))

        start_step = self._backend_instance.step
        end_step = int((end_time - self._start_time)/self._parameters.dt)
        num_step = max(0, end_step - start_step)

        progress = get_progress(progress_type)
        with progress(num_step) as prog_bar:
            while self._time(self._backend_instance.step) < __end_time:
                step, state = self._backend_instance.compute_step()
                self._dynamics.add(self._time(step), state.reshape(dim, dim))
                prog_bar.update(self._backend_instance.step - start_step)
            prog_bar.update(self._backend_instance.step - start_step)

        return self._dynamics

    def get_dynamics(self) -> Dynamics:
        """Returns the instance of Dynamics associated with the Tempo object.
        """
        return self._dynamics


def _analyse_correlation(
        corr_func: Callable[[np.ndarray],np.ndarray],
        times: np.ndarray,
        corr_vals: np.ndarray):
    """Check correlation function on a finer grid."""
    additional_times = (times[:-1] + times[1:])/2.0
    additional_corr_vals = corr_func(additional_times)
    new_times = list(times)
    new_corr_vals = list(corr_vals)
    for i in range(len(additional_times)):
        new_times.insert(2*i+1,additional_times[i])
        new_corr_vals.insert(2*i+1,additional_corr_vals[i])

    errors = []
    integrals = []
    integral = 0.0

    for i in range(len(times)-1):
        dt = new_times[2*i+2] - new_times[2*i]

        rough_int = 0.5 * dt * (new_corr_vals[2*i] + new_corr_vals[2*i+2])
        fine_int = 0.5 * (rough_int + dt * new_corr_vals[2*i+1])
        error = np.abs(rough_int-fine_int)
        errors.append(error)

        rough_abs_int = 0.5 * dt \
                * (np.abs(new_corr_vals[2*i]) + np.abs(new_corr_vals[2*i+2]))
        fine_abs_int = 0.5 * (rough_abs_int + dt * np.abs(new_corr_vals[2*i+1]))
        integral += fine_abs_int
        integrals.append(integral)

    full_abs_integral = integrals[-1]

    new_times = np.array(new_times)
    new_corr_val = np.array(new_corr_vals)
    errors = np.array(errors) / full_abs_integral
    integrals = np.array(integrals) / full_abs_integral

    return new_times, new_corr_val, errors, integrals

def _estimate_epsrel(
        dkmax: int,
        tollerance: float) -> float:
    """Heuristic estimation of appropriate epsrel for TEMPO."""
    power = np.log(dkmax)/np.log(4)-np.log(tollerance)/np.log(10)
    return np.power(10,-power)

GUESS_WARNING_MSG = "Estimating parameters for TEMPO computation. " \
    + "No guarantie that resulting TEMPO computation converges towards " \
    + "the correct dynamics! " \
    + "Please refere to the TEMPO documentation and check convergence by " \
    + "varying the parameters for TEMPO manually."

MAX_DKMAX_WARNING_MSG = f"Reached maximal recommended `dkmax` ({MAX_DKMAX})! " \
    + "Interrupt TEMPO parameter estimation. "\
    + "Please choose a lower tollerance, or analyse the correlation function " \
    + "to choose TEMPO parameters manually. " \
    + "Could not reach specified tollerance! "

def guess_tempo_parameters(
        bath: Bath,
        start_time: float,
        end_time: float,
        system: Optional[BaseSystem] = None,
        tollerance: Optional[float] = DEFAULT_TOLLERANCE) -> TempoParameters:
    """
    Function to roughly estimate appropriate parameters for a TEMPO
    computation.

    .. warning::

        No guarantie that resulting TEMPO calculation converges towards the
        correct dynamics! Please refere to the TEMPO documentation and check
        convergence by varying the parameters for TEMPO manually.

    Parameters
    ----------
    bath: Bath
        The bath.
    start_time: float
        The start time.
    end_time: float
        The time to which the TEMPO should be computed.
    system: BaseSystem
        The system.
    tollerance: float
        Tollerance for the parameter estimation.

    Returns
    -------
    tempo_parameters : TempoParameters
        Estimate of appropropriate tempo parameters.
    """
    assert isinstance(bath, Bath), \
        "Argument 'bath' must be a time_evolving_mpo.Bath object."
    try:
        __start_time = float(start_time)
        __end_time = float(end_time)
    except Exception as e:
        raise AssertionError("Start and end time must be a float.") from e
    if __end_time <= __start_time:
        raise ValueError("End time must be bigger than start time.")
    assert isinstance(system, (type(None), BaseSystem)), \
        "Argument 'system' must be 'None' or a time_evolving_mpo.BaseSystem object."
    try:
        __tollerance = float(tollerance)
    except Exception as e:
        raise AssertionError("Argument 'tollerance' must be float.") from e
    assert __tollerance > 0.0, \
        "Argument 'tollerance' must be larger then 0."
    warnings.warn(GUESS_WARNING_MSG, UserWarning)
    print("WARNING: "+GUESS_WARNING_MSG, file=sys.stderr, flush=True)

    max_tau = __end_time - __start_time
    if bath.correlations.max_correlation_time is not None:
        max_tau = min([max_tau, bath.correlations.max_correlation_time])

    corr_func = np.vectorize(bath.correlations.correlation)
    new_times = np.linspace(0, max_tau, 11, endpoint=True)
    new_corr_vals = corr_func(new_times)
    times = new_times
    corr_vals = new_corr_vals

    while True:
        if len(new_times) > MAX_DKMAX:
            warnings.warn(MAX_DKMAX_WARNING_MSG, UserWarning)
            break
        times = new_times
        corr_vals = new_corr_vals
        new_times, new_corr_vals, errors, integrals = \
                _analyse_correlation(corr_func, times, corr_vals)
        cut = np.where(integrals>(1-tollerance))[0][0]
        cut = cut+2 if cut+2<=len(times) else len(times)
        times = times[:cut]
        corr_vals = corr_vals[:cut]
        new_times = new_times[:2*cut-1]
        new_corr_vals = new_corr_vals[:2*cut-1]
        if (errors < tollerance).all():
            break

    dt = np.min(times[1:] - times[:-1])
    dkmax = len(times)
    epsrel = _estimate_epsrel(dkmax, tollerance)
    sys.stderr.flush()

    return TempoParameters(
        dt=dt,
        dkmax=dkmax,
        epsrel=epsrel,
        name="Roughly estimated parameters",
        description="Estimated with 'guess_tempo_parameters()'",
        description_dict={"tollerance":tollerance})


def tempo_compute(
        system: BaseSystem,
        bath: Bath,
        initial_state: ndarray,
        start_time: float,
        end_time: float,
        parameters: Optional[TempoParameters] = None,
        tollerance: Optional[float] = DEFAULT_TOLLERANCE,
        backend: Optional[Text] = None,
        backend_config: Optional[Dict] = None,
        progress_type: Optional[Text] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
        description_dict: Optional[Dict] = None) -> Dynamics:
    """
    Shortcut for creating a Tempo object and runing the computation.

    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The Bath (includes the coupling operator to the sytem).
    initial_state: ndarray
        The initial density matrix of the sytem.
    start_time: float
        The start time.
    end_time: float
        The time to which the TEMPO should be computed.
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    tollerance: float
        Tollerance for the parameter estimation (only applicable if
        `parameters` is None).
    backend: str (default = None)
        The name of the backend to use for the computation. If `backend` is
        ``None`` then the default backend is used.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``'silent'``, ``'simple'``, ``'bar'``}.  If `None` then
        the default progress type is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    description_dict: dict (default = None)
        An optional dictionary with descriptive data.
    """
    if parameters is None:
        assert tollerance is not None, \
            "If 'parameters' is 'None' then 'tollerance' must be " \
            + "a positive float."
        parameters = guess_tempo_parameters(bath=bath,
                                            start_time=start_time,
                                            end_time=end_time,
                                            system=system,
                                            tollerance=tollerance)
    tempo = Tempo(system,
                  bath,
                  parameters,
                  initial_state,
                  start_time,
                  backend,
                  backend_config,
                  name,
                  description,
                  description_dict)
    tempo.compute(end_time, progress_type=progress_type)
    return tempo.get_dynamics()

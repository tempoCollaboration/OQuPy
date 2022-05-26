# Copyright 2022 The TEMPO Collaboration
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
algorithm. This module is based on [Strathearn2017] and [Strathearn2018].

**[Strathearn2017]**
A. Strathearn, B.W. Lovett, and P. Kirton, *Efficient real-time path integrals
for non-Markovian spin-boson models*. New Journal of Physics, 19(9),
p.093009 (2017).

**[Strathearn2018]**
A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).
"""

import sys
from typing import Callable, Dict, Optional, Text, Tuple, Union
import warnings
from copy import copy

import numpy as np
from numpy import ndarray
from scipy.linalg import expm
from scipy import integrate

from oqupy.bath import Bath
from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype, MAX_DKMAX, DEFAULT_TOLERANCE
from oqupy.config import INTEGRATE_EPSREL, SUBDIV_LIMIT
from oqupy.config import TEMPO_BACKEND_CONFIG
from oqupy.correlations import BaseCorrelations
from oqupy.dynamics import Dynamics, DynamicsWithField
from oqupy.operators import commutator, acommutator
from oqupy.system import BaseSystem, System, TimeDependentSystem,\
    TimeDependentSystemWithField
from oqupy.backends.tempo_backend import TempoBackend
from oqupy.backends.tempo_backend import TempoWithFieldBackend
from oqupy.util import get_progress


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
        necessary memory time :math:`\tau_\mathrm{cut}`.
    epsrel: float
        The maximal relative error in the singular value truncation (done
        in the underlying tensor network algorithm). - It must be small enough
        such that the numerical compression (using tensor network algorithms)
        does not truncate relevant correlations.
    add_correlation_time: float
        Additional correlation time to include in the last influence
        functional as explained in [Strathearn2017].
    name: str (default = None)
        An optional name for the tempo parameters object.
    description: str (default = None)
        An optional description of the tempo parameters object.
    """
    def __init__(
            self,
            dt: float,
            dkmax: int,
            epsrel: float,
            add_correlation_time: Optional[float] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a TempoParameters object."""
        self.dt = dt
        self.dkmax = dkmax
        self.epsrel = epsrel
        self.add_correlation_time = add_correlation_time
        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dt                   = {} \n".format(self.dt))
        ret.append("  dkmax                = {} \n".format(self.dkmax))
        ret.append("  epsrel               = {} \n".format(self.epsrel))
        ret.append("  add_correlation_time = {} \n".format(
            self.add_correlation_time))
        return "".join(ret)

    @property
    def dt(self) -> float:
        """Length of a time step."""
        return self._dt

    @dt.setter
    def dt(self, new_dt: float) -> None:
        try:
            tmp_dt = float(new_dt)
        except Exception as e:
            raise AssertionError("Argument 'dt' must be float.") from e
        assert tmp_dt > 0.0, \
            "Argument 'dt' must be bigger than 0."
        self._dt = tmp_dt

    @property
    def dkmax(self) -> float:
        """Number of time steps that should be included in the non-Markovian
        memory. """
        return self._dkmax

    @dkmax.setter
    def dkmax(self, new_dkmax: float) -> None:
        try:
            if new_dkmax is None:
                tmp_dkmax = None
            else:
                tmp_dkmax = int(new_dkmax)
        except Exception as e:
            raise AssertionError("Argument 'dkmax' must be int or None.") \
                from e
        assert tmp_dkmax is None or tmp_dkmax > 0, \
            "Argument 'dkmax' must be bigger than or equal to 0 or None."
        self._dkmax = tmp_dkmax

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
            tmp_epsrel = float(new_epsrel)
        except Exception as e:
            raise AssertionError("Argument 'epsrel' must be float.") from e
        assert tmp_epsrel > 0.0, \
            "Argument 'epsrel' must be bigger than 0."
        self._epsrel = tmp_epsrel

    @property
    def add_correlation_time(self) -> float:
        """
        Additional correlation time to include in the last influence
        functional.
        """
        return self._add_correlation_time

    @add_correlation_time.setter
    def add_correlation_time(self, new_tau: Optional[float] = None) -> None:
        if new_tau is None:
            del self.add_correlation_time
        else:
            # check input: cutoff
            try:
                tmp_new_tau = float(new_tau)
            except Exception as e:
                raise AssertionError( \
                    "Additional correlation time must be a float.") from e
            if tmp_new_tau < 0:
                raise ValueError(
                    "Additional correlation time must be non-negative.")
            self._add_correlation_time = tmp_new_tau

    @add_correlation_time.deleter
    def add_correlation_time(self) -> None:
        self._add_correlation_time = None

class BaseTempo(BaseAPIClass):
    """
    Base class for all TEMPO objects.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    initial_state: ndarray
        The initial density matrix of the system.
    start_time: float
        The start time.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    """
    def __init__(
            self,
            bath: Bath,
            parameters: TempoParameters,
            initial_state: ndarray,
            start_time: float,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a BaseTempo object. """

        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath

        self._correlations = self._bath.correlations

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        try:
            tmp_initial_state = np.array(initial_state, dtype=NpDtype)
            tmp_initial_state.setflags(write=False)
        except Exception as e:
            raise AssertionError("Initial state must be numpy array.") from e
        assert len(tmp_initial_state.shape) == 2, \
            "Initial state is not a matrix."
        assert tmp_initial_state.shape[0] == \
            tmp_initial_state.shape[1], \
            "Initial state is not a square matrix."
        self._initial_state = tmp_initial_state
        self._dimension = self._initial_state.shape[0]

        try:
            tmp_start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = tmp_start_time

        if backend_config is None:
            self._backend_config = TEMPO_BACKEND_CONFIG
        else:
            self._backend_config = backend_config

        assert self._bath.dimension == self._dimension, \
            "Hilbertspace dimensions are unequal: " \
            + "initial state ({}), ".format(self._dimension) \
            + "and bath coupling ({}).".format(self._bath.dimension)

        super().__init__(name, description)

        tmp_coupling_comm = commutator(self._bath._coupling_operator)
        tmp_coupling_acomm = acommutator(self._bath._coupling_operator)
        self._coupling_comm = tmp_coupling_comm.diagonal()
        self._coupling_acomm = tmp_coupling_acomm.diagonal()

        self._dynamics = None
        self._backend_instance = None

    def _influence(self, dk: int) -> ndarray:
        """Create the influence functional matrix for a time step distance
        of dk. """
        return influence_matrix(
            dk,
            parameters=self._parameters,
            correlations=self._correlations,
            coupling_acomm=self._coupling_acomm,
            coupling_comm=self._coupling_comm)

    def _time(self, step: int) -> float:
        """Return the time that corresponds to the time step `step`. """
        return self._start_time + float(step)*self._parameters.dt

    def _get_num_step(self,
            start_step: int,
            end_time: float) -> Tuple[int, int]:
        """Return the number of steps required from start_step to reach
        end_time"""
        end_step = int((end_time - self._start_time)/self._parameters.dt)
        num_step = max(0, end_step - start_step)
        return num_step

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension. """
        return copy(self._dimension)

class Tempo(BaseTempo):
    """
    Class representing the entire TEMPO tensornetwork as introduced in
    [Strathearn2018].

    Parameters
    ----------
    system: System or TimeDependentSystem
        The system.
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    initial_state: ndarray
        The initial density matrix of the system.
    start_time: float
        The start time.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    """
    def __init__(
            self,
            system: Union[System, TimeDependentSystem],
            bath: Bath,
            parameters: TempoParameters,
            initial_state: ndarray,
            start_time: float,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a Tempo object. """

        assert isinstance(system, BaseSystem), \
            "Argument 'system' must be an instance of BaseSystem."
        self._system = system

        super().__init__(
                bath,
                parameters,
                initial_state,
                start_time,
                backend_config,
                name,
                description)

        assert self._system.dimension == self._dimension, \
            "Hilbertspace dimensions are unequal: " \
            + "system ({}), ".format(self._system.dimension) \
            + "initial state ({}), ".format(self._dimension) \
            + "and bath coupling ({}), ".format(self._bath.dimension)

        self._prepare_backend()

    def _prepare_backend(self):
        """Create and initialize the TEMPO backend. """
        dim = self._dimension
        initial_state = self._initial_state.reshape(dim**2)
        influence = self._influence
        unitary_transform = self._bath.unitary_transform
        propagators = self._propagators
        sum_north = np.array([1.0]*(dim**2))
        sum_west = np.array([1.0]*(dim**2))
        dkmax = self._parameters.dkmax
        epsrel = self._parameters.epsrel
        self._backend_instance = TempoBackend(
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
        self._dynamics = Dynamics(name=name,
                                  description=description)

    def _propagators(self, step: int):
        """Create the system propagators (first and second half) for the time
        step `step`. """
        dt = self._parameters.dt
        if isinstance(self._system, TimeDependentSystem):
            t = self._time(step)
            first_step = expm(self._system.liouvillian(t+dt/4.0)*dt/2.0)
            second_step = expm(self._system.liouvillian(t+dt*3.0/4.0)*dt/2.0)
        else:
            first_step = expm(self._system.liouvillian()*dt/2.0)
            second_step = expm(self._system.liouvillian()*dt/2.0)
        return first_step, second_step

    def compute(
            self,
            end_time: float,
            progress_type: Text = None) -> Dynamics:
        """
        Propagate (or continue to propagate) the TEMPO tensor network to
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
        tmp_end_time = _check_time(end_time)

        dim = self._dimension
        if self._backend_instance.step is None:
            step, state = self._backend_instance.initialize()
            self._init_dynamics()
            self._dynamics.add(self._time(step), state.reshape(dim, dim))

        start_step = self._backend_instance.step
        num_step = self._get_num_step(start_step, tmp_end_time)

        progress = get_progress(progress_type)
        title = "--> TEMPO computation:"
        with progress(num_step, title) as prog_bar:
            for i in range(num_step):
                prog_bar.update(i)
                step, state = self._backend_instance.compute_step()
                self._dynamics.add(self._time(step), state.reshape(dim, dim))
            prog_bar.update(num_step)

        return self._dynamics

    def get_dynamics(self) -> Dynamics:
        """Returns the instance of Dynamics associated with the Tempo object.
        """
        return self._dynamics

class TempoWithField(BaseTempo):
    r"""
    Class representing the TEMPO tensornetwork with coherent field
    evolution as introduced in [FowlerWright2021].

    Parameters
    ----------
    system: TimeDependentSystemWithField
        The (time-dependent) system with a coherent field.
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    initial_state: ndarray
        The initial density matrix of the system.
    initial_field: complex
        The initial field value.
    start_time: float
        The start time.
    subdiv_limit: int (default = config.SUBDIV_LIMIT)
        The maximum number of subdivisions used during the adaptive
        algorithm when integrating the system Liouvillian. If None
        then the Liouvillian is not integrated but sampled twice to
        to construct the system propagators at each timestep.
    epsrel: float (default = config.INTEGRATE_EPSREL)
        The relative error tolerance for the adaptive algorithm
        when integrating the system Liouvillian.
    backend_config: dict (default = None)
        The configuration of the backend. If `backend_config` is
        ``None`` then the default backend configuration is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    """
    def __init__(
            self,
            system: TimeDependentSystemWithField,
            bath: Bath,
            parameters: TempoParameters,
            initial_state: ndarray,
            initial_field: complex,
            start_time: float,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            epsrel: Optional[float] = INTEGRATE_EPSREL,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a TempoWithField object. """
        assert isinstance(system, TimeDependentSystemWithField), \
            "Argument 'system' must be an instance of " \
            "TimeDependentSystemWithField."
        self._system = system

        super().__init__(
                bath,
                parameters,
                initial_state,
                start_time,
                backend_config,
                name,
                description)

        try:
            tmp_field = complex(initial_field)
        except Exception as e:
            raise AssertionError("Argument `initial_field` must be "\
                    " complex.") from e
        self._initial_field = tmp_field

        try:
            tmp_epsrel = float(epsrel)
        except Exception as e:
            raise AssertionError("Argument `epsrel` must be float.") from e
        self.epsrel = tmp_epsrel

        if subdiv_limit is not None:
            try:
                tmp_subdiv_limit = int(subdiv_limit)
            except Exception as e:
                raise AssertionError("Argument `subdiv_limit` must be "\
                        "int.") from e
        else:
            tmp_subdiv_limit = None
        self.subdiv_limit = tmp_subdiv_limit

        self._prepare_backend()

    def _prepare_backend(self):
        """Create and initialize the TempoWithField backend. """
        dim = self._dimension
        initial_state = self._initial_state.reshape(dim**2)
        initial_field = self._initial_field
        influence = self._influence
        unitary_transform = self._bath.unitary_transform
        propagators = self._propagators
        compute_field = self._compute_field
        sum_north = np.array([1.0]*(dim**2))
        sum_west = np.array([1.0]*(dim**2))
        dkmax = self._parameters.dkmax
        epsrel = self._parameters.epsrel
        self._backend_instance = TempoWithFieldBackend(
                initial_state,
                initial_field,
                influence,
                unitary_transform,
                propagators,
                compute_field,
                sum_north,
                sum_west,
                dkmax,
                epsrel,
                config=self._backend_config)

    def _init_dynamics(self):
        """Create a DynamicsWithField object with metadata from the
        TempoWithField object. """
        name = None
        description = "computed from '{}' TempoWithField".format(self.name)
        self._dynamics = DynamicsWithField(name=name,
                                  description=description)

    def _propagators(self, step: int, state: ndarray, field: complex):
        """Create the system propagators (first and second half) for the time
        step `step`.

        Samples system Liouvillian if `self.subdiv_limit` is None,
        otherwise integrates this operator numerically."""
        dt = self._parameters.dt
        t0 = self._time(step)
        # SAMPLE
        if self.subdiv_limit is None:
            first_step = expm(
                    self._system.liouvillian(
                        t0, t0+dt/4.0, state, field)*dt/2.0)
            second_step = expm(
                    self._system.liouvillian(
                        t0, t0+dt*3.0/4.0, state, field)*dt/2.0)
            return first_step, second_step
        # ADAPTIVE
        liouvillian = lambda t: self._system.liouvillian(t0, t, state, field)
        first_step = expm(integrate.quad_vec(f=liouvillian,
                                             a=t0,
                                             b=t0+dt/2.0,
                                             epsrel=self.epsrel,
                                             limit=self.subdiv_limit)[0])
        second_step = expm(integrate.quad_vec(f=liouvillian,
                                             a=t0+dt/2.0,
                                             b=t0+dt,
                                             epsrel=self.epsrel,
                                             limit=self.subdiv_limit)[0])
        return first_step, second_step

    def _compute_field(self, step:int, state: ndarray, field: complex,
            next_state: Optional[ndarray] = None):
        r"""Compute the field value for the time step `step`. """
        dt = self._parameters.dt
        t = self._time(step)
        field_eom = self._system.field_eom
        state = state.reshape((self._dimension, self._dimension))
        rk1 = field_eom(t, state, field)
        if next_state is None:
            return rk1 * dt
        next_state = next_state.reshape((self._dimension, self._dimension))
        rk2 = field_eom(t + dt, next_state, field + rk1 * dt)
        return field + dt * (rk1 + rk2) / 2

    def compute(
            self,
            end_time: float,
            progress_type: Text = None) -> DynamicsWithField:
        """
        Propagate (or continue to propagate) the TEMPO tensor network and
        coherent field to time `end_time`.

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
        dynamics: DynamicsWithFields
            The instance of DynamicsWithField associated with the
            TempoWithField object.
        """

        tmp_end_time = _check_time(end_time)

        dim = self._dimension
        if self._backend_instance.step is None:
            step, state, field = self._backend_instance.initialize()
            self._init_dynamics()
            self._dynamics.add(self._time(step), state.reshape(dim, dim), field)

        start_step = self._backend_instance.step
        num_step = self._get_num_step(start_step, tmp_end_time)

        progress = get_progress(progress_type)
        title = "--> TEMPO-with-field computation:"

        with progress(num_step, title) as prog_bar:
            for i in range(num_step):
                prog_bar.update(i)
                step, state, field = self._backend_instance.compute_step()
                self._dynamics.add(
                    self._time(step), state.reshape(dim, dim), field)
            prog_bar.update(num_step)

        return self._dynamics

    def get_dynamics(self) -> DynamicsWithField:
        """Returns DynamicsWithField instance associated with the Tempo object.
        """
        return self._dynamics

def _check_time(end_time):
    """input check on end time of a tempo computation"""
    try:
        tmp_end_time = float(end_time)
    except Exception as e:
        raise AssertionError("End time must be a float.") from e
    return tmp_end_time

def influence_matrix(
        dk: int,
        parameters: TempoParameters,
        correlations: BaseCorrelations,
        coupling_acomm: ndarray,
        coupling_comm: ndarray):
    """Compute the influence functional matrix. """
    dt = parameters.dt
    dkmax = parameters.dkmax

    if dk == 0:
        time_1 = 0.0
        time_2 = None
        shape = "upper-triangle"
    elif dk < 0:
        time_1 = float(dkmax) * dt
        if parameters.add_correlation_time is not None:
            time_2 = float(dkmax) * dt \
                + np.min([float(-dk) * dt,
                            1.0*dt + parameters.add_correlation_time])
        else:
            return None
        shape = "rectangle"
    else:
        time_1 = float(dk) * dt
        time_2 = None
        shape = "square"

    eta_dk = correlations.correlation_2d_integral( \
        delta=dt,
        time_1=time_1,
        time_2=time_2,
        shape=shape,
        epsrel=parameters.epsrel)
    op_p = coupling_acomm
    op_m = coupling_comm

    if dk == 0:
        infl = np.diag(np.exp(-op_m*(eta_dk.real*op_m \
                                        + 1j*eta_dk.imag*op_p)))
    else:
        infl = np.exp(-np.outer(eta_dk.real*op_m \
                                + 1j*eta_dk.imag*op_p, op_m))

    return infl

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
        tolerance: float) -> float:
    """Heuristic estimation of appropriate epsrel for TEMPO."""
    power = np.log(dkmax)/np.log(4)-np.log(tolerance)/np.log(10)
    return np.power(10,-power)

GUESS_WARNING_MSG = "Estimating parameters for TEMPO computation. " \
    + "No guarantee that resulting TEMPO computation converges towards " \
    + "the correct dynamics! " \
    + "Please refer to the TEMPO documentation and check convergence by " \
    + "varying the parameters for TEMPO manually."

MAX_DKMAX_WARNING_MSG = f"Reached maximal recommended `dkmax` ({MAX_DKMAX})! " \
    + "Interrupt TEMPO parameter estimation. "\
    + "Please choose a lower tolerance, or analyse the correlation function " \
    + "to choose TEMPO parameters manually. " \
    + "Could not reach specified tolerance! "

def guess_tempo_parameters(
        bath: Bath,
        start_time: float,
        end_time: float,
        system: Optional[BaseSystem] = None,
        tolerance: Optional[float] = DEFAULT_TOLERANCE) -> TempoParameters:
    """
    Function to roughly estimate appropriate parameters for a TEMPO
    computation.

    .. warning::

        No guarantee that resulting TEMPO calculation converges towards the
        correct dynamics! Please refer to the TEMPO documentation and check
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
    tolerance: float
        Tolerance for the parameter estimation.

    Returns
    -------
    tempo_parameters : TempoParameters
        Estimate of appropriate tempo parameters.
    """
    assert isinstance(bath, Bath), \
        "Argument 'bath' must be a oqupy.Bath object."
    try:
        tmp_start_time = float(start_time)
        tmp_end_time = float(end_time)
    except Exception as e:
        raise AssertionError("Start and end time must be a float.") from e
    if tmp_end_time <= tmp_start_time:
        raise ValueError("End time must be bigger than start time.")
    assert isinstance(system, (type(None), BaseSystem)), \
        "Argument 'system' must be 'None' or a oqupy.BaseSystem object."
    try:
        tmp_tolerance = float(tolerance)
    except Exception as e:
        raise AssertionError("Argument 'tolerance' must be float.") from e
    assert tmp_tolerance > 0.0, \
        "Argument 'tolerance' must be larger then 0."
    warnings.warn(GUESS_WARNING_MSG, UserWarning)
    print("WARNING: "+GUESS_WARNING_MSG, file=sys.stderr, flush=True)

    max_tau = tmp_end_time - tmp_start_time

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
        cut = np.where(integrals>(1-tolerance))[0][0]
        cut = cut+2 if cut+2<=len(times) else len(times)
        times = times[:cut]
        corr_vals = corr_vals[:cut]
        new_times = new_times[:2*cut-1]
        new_corr_vals = new_corr_vals[:2*cut-1]
        if (errors < tolerance).all():
            break

    dt = np.min(times[1:] - times[:-1])
    dkmax = len(times)
    epsrel = _estimate_epsrel(dkmax, tolerance)
    sys.stderr.flush()

    return TempoParameters(
        dt=dt,
        dkmax=dkmax,
        epsrel=epsrel,
        name="Roughly estimated parameters",
        description="Estimated with 'guess_tempo_parameters()'")


def tempo_compute(
        system: BaseSystem,
        bath: Bath,
        initial_state: ndarray,
        start_time: float,
        end_time: float,
        parameters: Optional[TempoParameters] = None,
        tolerance: Optional[float] = DEFAULT_TOLERANCE,
        backend_config: Optional[Dict] = None,
        progress_type: Optional[Text] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None) -> Dynamics:
    """
    Shortcut for creating a Tempo object and running the computation.

    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The Bath (includes the coupling operator to the system).
    initial_state: ndarray
        The initial density matrix of the system.
    start_time: float
        The start time.
    end_time: float
        The time to which the TEMPO should be computed.
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    tolerance: float
        Tolerance for the parameter estimation (only applicable if
        `parameters` is None).
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
    """
    if parameters is None:
        assert tolerance is not None, \
            "If 'parameters' is 'None' then 'tolerance' must be " \
            + "a positive float."
        parameters = guess_tempo_parameters(bath=bath,
                                            start_time=start_time,
                                            end_time=end_time,
                                            system=system,
                                            tolerance=tolerance)
    tempo = Tempo(system,
                  bath,
                  parameters,
                  initial_state,
                  start_time,
                  backend_config,
                  name,
                  description)
    tempo.compute(end_time, progress_type=progress_type)
    return tempo.get_dynamics()

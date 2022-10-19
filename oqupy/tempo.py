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
from typing import Callable, Dict, List, Optional, Text, Tuple, Union
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
from oqupy.dynamics import Dynamics, MeanFieldDynamics
from oqupy.operators import commutator, acommutator
from oqupy.system import BaseSystem, System, TimeDependentSystem,\
    TimeDependentSystemWithField, MeanFieldSystem
from oqupy.backends.tempo_backend import TempoBackend
from oqupy.backends.tempo_backend import MeanFieldTempoBackend
from oqupy.util import check_convert, check_isinstance, check_true,\
        get_progress


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

class Tempo(BaseAPIClass):
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
        super().__init__(name, description)

        self._system, self._initial_state, self._bath, self._dimension = \
            _tempo_physical_input_parse(False, system, initial_state, bath)

        self._correlations = self._bath.correlations

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        try:
            tmp_start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = tmp_start_time

        if backend_config is None:
            self._backend_config = TEMPO_BACKEND_CONFIG
        else:
            self._backend_config = backend_config

        tmp_coupling_comm = commutator(self._bath._coupling_operator)
        tmp_coupling_acomm = acommutator(self._bath._coupling_operator)
        self._coupling_comm = tmp_coupling_comm.diagonal()
        self._coupling_acomm = tmp_coupling_acomm.diagonal()

        self._dynamics = None
        self._backend_instance = None

        assert self._system.dimension == self._dimension, \
            "Hilbertspace dimensions are unequal: " \
            + "system ({}), ".format(self._system.dimension) \
            + "initial state ({}), ".format(self._dimension) \
            + "and bath coupling ({}), ".format(self._bath.dimension)

        self._prepare_backend()

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

class MeanFieldTempo(BaseAPIClass):
    r"""
    Class for the evolution of a collection of system
    (`TimeDependentSystemWithField`) within a mean-field system
    (`MeanFieldSystem` ) together with a coherent field coupled to the systems.
    Based on the TEMPO tensor network with field evolution introduced in
    [FowplerWright2021].

    Parameters
    ----------
    mean_field_system: MeanFieldSystem
        The `MeanFieldSystem` representing the collection of time-dependent
        systems and coherent field.
    bath_list: List[Bath]
        List of Bath objects, one for each system in the mean-field system.
    parameters: TempoParameters
        The parameters for the TEMPO computations. These are used by all
        systems in the mean-field system.
    initial_state_list: List[ndarray]
        List of initial density matrices, one for each system in the
        mean-field system.
    initial_field: complex
        The initial field value.
    start_time: float (default = 0.0)
        The start time.
    subdiv_limit: int (default = config.SUBDIV_LIMIT)
        The maximum number of subdivisions used during the adaptive
        algorithm when integrating each system Liouvillian. If None
        then the Liouvillian is not integrated but sampled twice to
        to construct the system propagators at a timestep.
    hamiltonian_epsrel: float (default = config.INTEGRATE_EPSREL)
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
            mean_field_system: MeanFieldSystem,
            bath_list: List[Bath],
            parameters: TempoParameters,
            initial_state_list: List[ndarray],
            initial_field: complex,
            start_time: Optional[float] = 0.0,
            subdiv_limit: Optional[int] = SUBDIV_LIMIT,
            hamiltonian_epsrel: Optional[float] = INTEGRATE_EPSREL,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a MeanFieldTempo object. """
        assert isinstance(mean_field_system, MeanFieldSystem), \
            "Argument 'mean_field_system' must be an instance of " \
            "MeanFieldSystem."
        self._mean_field_system = mean_field_system
        self._dynamics = None

        if backend_config is None:
            self._backend_config = TEMPO_BACKEND_CONFIG
        else:
            self._backend_config = backend_config

        # Parameters used for each tempo computation
        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        super().__init__(name, description)

        assert isinstance(bath_list, list), "bath_list must be a list of "\
                " Bath objects"
        assert isinstance(initial_state_list, list), "initial_state_list "\
                "must be a list of state matrices"

        assert len(mean_field_system.system_list) == len(bath_list) ==\
                len(initial_state_list),\
                    f"The lengths of bath_list ({len(bath_list)}) "\
                    f"and initial_state_list ({len(initial_state_list)}) must "\
                    f"match the number ({len(mean_field_system.system_list)}) "\
                    "of systems in mean_field_system."

        # List of tuples, one for each system: (system, initial_state,
        # bath, hs_dim)
        parsed_system_tuple_list = [_tempo_physical_input_parse(
            True, system, initial_state, bath)
            for system, initial_state, bath in zip(
                mean_field_system.system_list, initial_state_list,
                bath_list)]

        parsed_parameter_names = ["system", "initial_state", "bath", "hs_dim"]
        # Dictionary of keys from parsed_parameter_names. The items are
        # a lists of systems (Key = "system") and corresponding lists of
        # values (Key = "initial_state", "bath", "hs_dim"). Considered to be
        # 'parameters' for the overall (mean-field system dynamics) computation.
        self._parsed_parameters_dict = {}
        for i, parsed_parameter_name in enumerate(parsed_parameter_names):
            self._parsed_parameters_dict[parsed_parameter_name] = []
            for parsed_parameter_tuple in parsed_system_tuple_list:
                self._parsed_parameters_dict[parsed_parameter_name].append(
                    parsed_parameter_tuple[i])

        # Check initial field can be cast to complex, and start time float
        self._initial_field = check_convert(initial_field, complex,
                                            "initial_field")
        self._start_time = check_convert(start_time, float, "start_time")
        # Input checks on these parameters (used when commutating the
        # propagators for each system) are done in the setter functions below.
        # naming here to avoid conflict with parameters in self._parameters:w
        self._hamiltonian_epsrel = hamiltonian_epsrel
        self._subdiv_limit = subdiv_limit
        # Prepare the MeanFieldTempo backend
        self._prepare_backend()

    # These properties should move to TempoParameters when adaptive propagator
    # construction has been added to Tempo. It doesn't make sense to allow
    # these properties to be changed unless propagator_list is regenerated
    @property
    def hamiltonian_epsrel(self) -> float:
        """The epsrel used to construct system propagators by integration. """
        return self._hamiltonian_epsrel

    @hamiltonian_epsrel.setter
    def hamiltonian_epsrel(self, new_hamiltonian_epsrel: float) -> None:
        try:
            tmp_hamiltonian_epsrel = float(new_hamiltonian_epsrel)
        except Exception as e:
            raise AssertionError("Argument 'hamiltonian_epsrel' must"\
                    " be float.") from e
        assert tmp_hamiltonian_epsrel > 0.0, \
            "Argument 'hamiltonian_epsrel' must be positive."
        self._hamiltonian_epsrel = tmp_hamiltonian_epsrel

    @property
    def subdiv_limit(self) -> Union[float, None]:
        """The subdiv_limit used to construct system propagators. """
        return self._subdiv_limit

    @subdiv_limit.setter
    def subdiv_limit(self, new_subdiv_limit: float) -> None:
        if new_subdiv_limit is None:
            self._subdiv_limit = None
            return
        try:
            tmp_subdiv_limit = float(new_subdiv_limit)
        except Exception as e:
            raise AssertionError("Argument 'subdiv_limit' must be float.")\
                    from e
        assert tmp_subdiv_limit > 0, \
            "Argument 'subdiv_limit' must be positive."
        self._subdiv_limit = tmp_subdiv_limit

    def _prepare_backend(self):
        """Create and initialize the MeanFieldTempo backend. """
        initial_state_list = \
                self._parsed_parameters_dict["initial_state"]
        initial_field = self._initial_field
        influence_list = [self._get_influence(bath)
                for bath in self._parsed_parameters_dict["bath"]]
        unitary_transform_list = [bath.unitary_transform
                for bath in self._parsed_parameters_dict["bath"]]
        propagators_list = [self._get_propagators(system)
            for system in self._parsed_parameters_dict["system"]]
        compute_field = self._compute_field
        compute_field_derivative = self._compute_field_derivative
        sum_north_list = [np.array([1.0]*(dim**2))
                for dim in self._parsed_parameters_dict["hs_dim"]]
        sum_west_list = [np.array([1.0]*(dim**2))
                for dim in self._parsed_parameters_dict["hs_dim"]]
        # N.B. For now all baths constrained to have same memory length
        dkmax = self._parameters.dkmax
        epsrel = self._parameters.epsrel
        self._backend_instance = MeanFieldTempoBackend(
                initial_state_list,
                initial_field,
                influence_list,
                unitary_transform_list,
                propagators_list,
                compute_field,
                compute_field_derivative,
                sum_north_list,
                sum_west_list,
                dkmax,
                epsrel,
                config=self._backend_config)

    def _init_dynamics(self):
        """Create a MeanFieldDynamics object with metadata from the
        MeanFieldTempo object. """
        name = None
        description = "computed from '{}' MeanFieldTempo".format(self.name)
        self._dynamics = MeanFieldDynamics(name=name,
                                  description=description)

    def _get_propagators(self, system):
        """Prepare propagator functions for a system according to subdiv_limit.
        """
        # SAMPLE
        if self._subdiv_limit is None:
            def propagators(step: int, field: complex,
                    field_derivative: complex):
                dt = self._parameters.dt
                t = self._time(step)
                first_step = expm(system.liouvillian(t, t+dt/4.0,
                    field, field_derivative)*dt/2.0)
                second_step = expm(system.liouvillian(t, t+dt*3.0/4.0,
                    field, field_derivative)*dt/2.0)
                return first_step, second_step
        # ADAPTIVE
        else:
            def propagators(step: int, field:complex,
                    field_derivative: complex):
                dt = self._parameters.dt
                t = self._time(step)
                liouvillian = lambda tau: system.liouvillian(t, tau,
                        field, field_derivative)
                first_step = expm(integrate.quad_vec(f=liouvillian,
                                             a=t,
                                             b=t+dt/2.0,
                                             epsrel=self._hamiltonian_epsrel,
                                             limit=self._subdiv_limit)[0])
                second_step = expm(integrate.quad_vec(f=liouvillian,
                                             a=t+dt/2.0,
                                             b=t+dt,
                                             epsrel=self._hamiltonian_epsrel,
                                             limit=self._subdiv_limit)[0])
                return first_step, second_step
        return propagators

    def _get_influence(self, bath) -> Callable[[int], ndarray]:
        """Create function that calculates the influence functional
        matrix for a bath. """
        coupling_comm = commutator(bath.coupling_operator).diagonal()
        coupling_acomm = acommutator(bath.coupling_operator).diagonal()
        def influence(dk: int) -> ndarray:
            return influence_matrix(
                dk,
                parameters=self._parameters,
                correlations=bath.correlations,
                coupling_acomm=coupling_acomm,
                coupling_comm=coupling_comm)
        return influence

    def _compute_field(self, step:int, state_list: List[ndarray],
            field: complex, next_state_list: Optional[List[ndarray]] = None):
        r"""Compute the field value for the time step `step`. Uses 2nd
        order Runge-Kutta if next_state_list is provided. """
        dt = self._parameters.dt
        t = self._time(step)

        state_list = [state.reshape((hs_dim, hs_dim)) for state, hs_dim
                in zip(state_list, self._parsed_parameters_dict["hs_dim"])]

        # could equally call self._compute_field_derivative
        rk1 = self._mean_field_system.field_eom(t, state_list, field)
        if next_state_list is None:
            return rk1 * dt
        next_state_list = [state.reshape((hs_dim, hs_dim)) for state, hs_dim
                in zip(next_state_list, self._parsed_parameters_dict["hs_dim"])]
        #  perform second order Runge-Kutta calculation
        rk2 = self._mean_field_system.field_eom(t + dt, next_state_list,
                                                field + rk1 * dt)
        return field + dt * (rk1 + rk2) / 2

    def _compute_field_derivative(self, step:int, state_list: List[ndarray],
                                  field:complex):
        r"""Compute the field derivative for the time step `step`. """
        t = self._time(step)
        state_list = [state.reshape((hs_dim, hs_dim)) for state, hs_dim
                in zip(state_list, self._parsed_parameters_dict["hs_dim"])]
        return self._mean_field_system.field_eom(t, state_list, field)

    def compute(
            self,
            end_time: float,
            progress_type: Text = None) -> MeanFieldDynamics:
        """
        Propagate (or continue to propagate) the TEMPO tensor networks for
        each system and coherent field to time `end_time`.

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
        dynamics: MeanFieldDynamics
            The instance of `MeanFieldDynamics` describing each system
            dynamics and the field dynamics accounting for the interaction with
            the environment.
        """

        tmp_end_time = _check_time(end_time)

        if self._backend_instance.step is None:
            step, system_states, field = self._backend_instance.initialize()
            self._init_dynamics()
            self._dynamics.add(self._time(step), system_states, field)

        start_step = self._backend_instance.step
        num_step = self._get_num_step(start_step, tmp_end_time)

        progress = get_progress(progress_type)
        title = "--> TEMPO-with-field computation:"

        with progress(num_step, title) as prog_bar:
            for i in range(num_step):
                prog_bar.update(i)
                step, state_list, field = self._backend_instance.compute_step()
                matrix_list = [state.reshape((hs_dim, hs_dim)) for state, hs_dim
                    in zip(state_list, self._parsed_parameters_dict["hs_dim"])]
                self._dynamics.add(
                    self._time(step), matrix_list, field)
            prog_bar.update(num_step)

        return self._dynamics

    def get_dynamics(self) -> MeanFieldDynamics:
        """Returns the instance of MeanFieldDynamics associated with the
        tempo object.
        """
        return self._dynamics

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
    Cannot be used for mean-field Tempo.

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

def _tempo_physical_input_parse(
       with_field, system, initial_state, bath) -> tuple:
    if with_field:
        check_isinstance(
            system, TimeDependentSystemWithField, "system")
    else:
        check_isinstance(
            system, (System, TimeDependentSystem), "system")
    hs_dim = system.dimension

    check_isinstance(initial_state, ndarray, "initial_state")
    check_true(
        initial_state.shape == (hs_dim, hs_dim),
        "Initial sate must be a square matrix of " \
            + f"dimension {hs_dim}x{hs_dim}.")

    assert isinstance(bath, Bath), \
        "Argument 'bath' must be an instance of Bath."

    assert bath.dimension == hs_dim, \
            "Hilbertspace dimensions are unequal: " \
            + "system ({}), ".format(hs_dim) \
            + "and bath coupling ({}).".format(bath.dimension)

    parameters = (system, initial_state, bath, hs_dim)
    return parameters

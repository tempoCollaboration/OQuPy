# Copyright 2020 The TEMPO Collaboration
#
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
Module on for the original time evolving matrix product operator (TEMPO)
algorithm. This code is based on [Strathearn2018].

**[Strathearn2018]** Strathearn, A., Kirton, P., Kilda, D., Keeling, J.
and Lovett, B.W., 2018. *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators.* Nature communications, 9(1), pp.1-9.
"""

from typing import Dict, Optional, Text
from typing import Any as ArrayLike
import warnings

from copy import copy
from numpy import array, ndarray, diag, exp, outer
from scipy.linalg import expm

from time_evolving_mpo.backends.backend_factory import get_backend
from time_evolving_mpo.bath import Bath
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.config import NpDtype
from time_evolving_mpo.dynamics import Dynamics
from time_evolving_mpo.system import BaseSystem
from time_evolving_mpo.util import commutator, acommutator
from time_evolving_mpo.util import get_progress


class TempoParameters(BaseAPIClass):
    r"""
    Parameters for the TEMPO computation.

    .. todo::

        Explain effect of parameters in more detail.

    Parameters
    ----------
    dt: float
        Length of a time step :math:`\delta t`.
    dkmax: int
        Number of time steps :math:`K\in\mathbb{N}` that should be included in
        the non-Markovian memory.
    epsrel: float
        The maximal relative error in the singular value truncation (done
        in the underlying tensor network algorithm).
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
        except:
            raise AssertionError("Argument 'dt' must be float.")
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
        except:
            raise AssertionError("Argument 'dkmax' must be int or None.")
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
        except:
            raise AssertionError("Argument 'epsrel' must be float.")
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
        The name of the backend to use for the computation. If `backend` is
        ``None`` then the default backend is used.
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
            initial_state: ArrayLike,
            start_time: float,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a Tempo object. """
        self._backend = get_backend(backend, backend_config)

        assert isinstance(system, BaseSystem), \
            "Argument 'system' must be an instance of BaseSystem."
        self._system = system

        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        try:
            __initial_state = array(initial_state, dtype=NpDtype)
            __initial_state.setflags(write=False)
        except:
            raise AssertionError("Initial state must be numpy array")
        assert len(__initial_state.shape) == 2, \
            "Initial state is not a matrix."
        assert __initial_state.shape[0] == \
            __initial_state.shape[1], \
            "Initial state is not a square matrix."
        self._initial_state = __initial_state
        self._dimension = self._initial_state.shape[0]

        try:
            __start_time = float(start_time)
        except:
            raise AssertionError("Start time must be a float.")
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

        self._init_tempo_backend()

    def _init_dynamics(self):
        """Create a Dynamics object with metadata from the Tempo object. """
        name = "computed from '{}' tempo".format(self.name)
        description = None
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
            "spectral_density_type":str(type(self._bath.spectral_density)),
            "spectral_density_name": \
                self._bath.spectral_density.name,
            "spectral_density_description": \
                self._bath.spectral_density.description,
            "spectral_density_description_dict": \
                self._bath.spectral_density.description_dict,
            "backend_type":str(type(self._backend)),
            "initial_state":self._initial_state,
            "dt":self._parameters.dt,
            "dkmax":self._parameters.dkmax,
            "epsrel":self._parameters.epsrel,
            }
        self._dynamics = Dynamics(name=name,
                                  description=description,
                                  description_dict=description_dict)

    def _init_tempo_backend(self):
        """Create and initialize the tempo backend. """
        dim = self._dimension
        initial_state = self._initial_state.reshape(dim**2)
        influence = self._influence
        propagators = self._propagators
        sum_north = array([1.0]*(dim**2))
        sum_west = array([1.0]*(dim**2))
        dkmax = self._parameters.dkmax
        epsrel = self._parameters.epsrel
        self._tempo_backend = \
            self._backend.get_tempo_backend(initial_state,
                                            influence,
                                            propagators,
                                            sum_north,
                                            sum_west,
                                            dkmax,
                                            epsrel)

    def _influence(self, dk: int):
        """Create the influence functional matrix for a time step distance
        of dk. """

        if dk == 0:
            shape = "upper-triangle"
        else:
            shape = "square"

        dt = self._parameters.dt
        eta_dk = self._bath.spectral_density.correlation_2d_integral( \
            time_1=float(dk)*dt,
            delta=dt,
            temperature=self._bath.temperature,
            shape=shape,
            epsrel=self._parameters.epsrel)
        op_p = self._coupling_acomm
        op_m = self._coupling_comm

        if dk == 0:
            infl = diag(exp(-op_m*(eta_dk.real*op_m + 1j*eta_dk.imag*op_p)))
        else:
            infl = exp(-outer(eta_dk.real*op_m + 1j*eta_dk.imag*op_p, op_m))

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

    def compute(self, end_time: float, progress_type: Text = None) -> None:
        """
        Propagate (or continue to propagete) the TEMPO tensor network to
        time `end_time`.

        Parameters
        ----------
        end_time: float
            The time to which the TEMPO should be computed.
        progress_type: str (default = None)
            The progress report type during the computation. Types are:
            {``silent``, ``simple`, ``bar``}. If `None` then
            the default progress type is used.
        """
        try:
            __end_time = float(end_time)
        except:
            raise AssertionError("End time must be a float.")

        dim = self._dimension
        if self._tempo_backend.step is None:
            step, state = self._tempo_backend.initialize()
            self._init_dynamics()
            self._dynamics.add(self._time(step), state.reshape(dim, dim))

        start_step = self._tempo_backend.step
        end_step = int((end_time - self._start_time)/self._parameters.dt)
        num_step = max(0, end_step - start_step)

        progress = get_progress(progress_type)
        with progress(num_step) as prog_bar:
            while self._time(self._tempo_backend.step) < __end_time:
                step, state = self._tempo_backend.compute_step()
                self._dynamics.add(self._time(step), state.reshape(dim, dim))
                prog_bar.update(self._tempo_backend.step - start_step)
            prog_bar.update(self._tempo_backend.step - start_step)

    def get_dynamics(self) -> Dynamics:
        """Returns a copy of the computed dynamics. """
        return copy(self._dynamics)


GUESS_WARNING_MSG = "Estimating parameters for TEMPO calculation. " \
    + "No guarantie that resulting TEMPO calculation converges towards " \
    + "the correct dynamics! " \
    + "Please refere to the TEMPO documentation and check convergence by " \
    + "varying the parameters for TEMPO manually."

PLACEHOLDER_MSG = "This is just a placeholder and not really implemented yet."
def guess_tempo_parameters(
        system: BaseSystem,
        bath: Bath,
        tollerance: float) -> TempoParameters:
    """
    Function to roughly estimate appropriate parameters for a TEMPO
    computation.

    .. warning::

        No guarantie that resulting TEMPO calculation converges towards the
        correct dynamics! Please refere to the TEMPO documentation and check
        convergence by varying the parameters for TEMPO manually.

    .. todo::

        This function is not really implemented yet.

    Parameters
    ----------
    system: BaseSystem
        The system.
    bath: Bath
        The bath.
    tollerance: float
        A measure for how exact/rough the computation should be.
    """
    assert isinstance(system, BaseSystem), \
        "Argument 'system' must be a time_evolving_mpo.BaseSystem object."
    assert isinstance(bath, Bath), \
        "Argument 'bath' must be a time_evolving_mpo.Bath object."
    try:
        __tollerance = float(tollerance)
    except:
        raise AssertionError("Argument 'tollerance' must be float.")
    assert __tollerance > 0.0, \
        "Argument 'tollerance' must be larger then 0."
    warnings.warn(GUESS_WARNING_MSG, UserWarning)
    pass # ToDo
    warnings.warn(PLACEHOLDER_MSG, UserWarning)
    return TempoParameters(
        dt=0.05,
        dkmax=20,
        epsrel=2**(-15),
        name="Roughly estimated parameters",
        description="Estimated with 'guess_tempo_parameters()'")


def tempo_compute(
        system: BaseSystem,
        bath: Bath,
        initial_state: ArrayLike,
        start_time: float,
        end_time: float,
        parameters: Optional[TempoParameters] = None,
        tollerance: Optional[float] = None,
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
    parameters: TempoParameters
        The parameters for the TEMPO computation.
    initial_state: ndarray
        The initial density matrix of the sytem.
    start_time: float
        The start time.
    end_time: float
        The time to which the TEMPO should be computed.
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
        parameters = guess_tempo_parameters(system, bath, tollerance)
    else:
        assert tollerance is None, \
            "If 'parameters' is given then 'tollerance' must be " \
            + "'None'."
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

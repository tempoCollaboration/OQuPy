# Copyright 2021 The TEMPO Collaboration
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
Module for the process tensor time evolving matrix product operator algorithm
(PT-TEMPO).
This code is based on [Strathearn2018, Pollock2018, Jorgensen2019, Fux2021].

**[Strathearn2018]**
A. Strathearn, P. Kirton, D. Kilda, J. Keeling and
B. W. Lovett,  *Efficient non-Markovian quantum dynamics using
time-evolving matrix product operators*, Nat. Commun. 9, 3322 (2018).

**[Pollock2018]**
F.  A.  Pollock,  C.  Rodriguez-Rosario,  T.  Frauenheim,
M. Paternostro, and K. Modi, *Non-Markovian quantumprocesses: Complete
framework and efficient characterization*, Phys. Rev. A 97, 012127 (2018).

**[Jorgensen2019]**
M. R. Jørgensen and F. A. Pollock, *Exploiting the causal tensor network
structure of quantum processes to efficiently simulate non-markovian path
integrals*, Phys. Rev. Lett. 123, 240602 (2019)

**[Fux2021]**
G. E. Fux, E. Butler, P. R. Eastham, B. W. Lovett, and
J. Keeling, *Efficient exploration of Hamiltonian parameter space for
optimal control of non-Markovian open quantum systems*, arXiv2101.03071
(2021).
"""

from typing import Dict, Optional, Text
from copy import copy

import numpy as np

from time_evolving_mpo.backends.backend_factory import get_pt_tempo_backend
from time_evolving_mpo.base_api import BaseAPIClass
from time_evolving_mpo.bath import Bath
# from time_evolving_mpo.config import NpDtype
# from time_evolving_mpo.config import PT_MAX_DKMAX, PT_DEFAULT_TOLLERANCE
from time_evolving_mpo.config import PT_DEFAULT_TOLLERANCE
from time_evolving_mpo.process_tensor import ProcessTensor
from time_evolving_mpo.tempo import TempoParameters
from time_evolving_mpo.tempo import guess_tempo_parameters
from time_evolving_mpo.util import commutator, acommutator
from time_evolving_mpo.util import get_progress


class PtTempoParameters(TempoParameters):
    r"""
    Parameters for the PT-TEMPO computation.

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
    pass # Inherit everything from TempoParameters


class PtTempo(BaseAPIClass):
    """
    Class to facilitate a PT-TEMPO computation.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the sytem).
    parameters: PtTempoParameters
        The parameters for the PT-TEMPO computation.
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
            bath: Bath,
            start_time: float,
            end_time: float,
            parameters: PtTempoParameters,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Create a PtTempo object. """
        self._backend_class, self._backend_config = \
            get_pt_tempo_backend(backend, backend_config)

        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath
        self._dimension = self._bath.dimension
        self._correlations = self._bath.correlations

        assert isinstance(parameters, PtTempoParameters), \
            "Argument 'parameters' must be an instance of PtTempoParameters."
        self._parameters = parameters

        try:
            __start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = __start_time

        try:
            __end_time = float(end_time)
        except Exception as e:
            raise AssertionError("End time must be a float.") from e
        self._end_time = __end_time

        super().__init__(name, description, description_dict)

        __coupling_comm = commutator(self._bath._coupling_operator)
        __coupling_acomm = acommutator(self._bath._coupling_operator)
        self._coupling_comm = __coupling_comm.diagonal()
        self._coupling_acomm = __coupling_acomm.diagonal()

        __num_steps = int((end_time - self._start_time)/self._parameters.dt)
        assert __num_steps >= 2, \
            "Parameter `end_time` must be more than two times steps " \
            + "larger than the parameter `start_time`!"
        self._num_steps = __num_steps

        self._backend_instance = None
        self._init_pt_tempo_backend()

    def _init_pt_tempo_backend(self):
        """Create and initialize the pt-tempo backend. """
        dimension = self._dimension
        influence = self._influence
        unitary_transform = self._bath.unitary_transform
        sum_north = np.array([1.0]*(dimension**2))
        sum_west = np.array([1.0]*(dimension**2))
        num_steps = self._num_steps
        dkmax = self._parameters.dkmax
        if dkmax is None:
            dkmax = num_steps
        epsrel = self._parameters.epsrel
        self._backend_instance = self._backend_class(
                dimension,
                influence,
                unitary_transform,
                sum_north,
                sum_west,
                num_steps,
                dkmax,
                epsrel,
                config=self._backend_config)

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

    # def _time(self, step: int):
    #     """Return the time that corresponds to the time step `step`. """
    #     return self._start_time + float(step)*self._parameters.dt

    @property
    def dimension(self) -> np.ndarray:
        """Hilbert space dimension. """
        return copy(self._dimension)

    def compute(self, progress_type: Optional[Text] = None) -> None:
        """
        Propagate (or continue to propagete) the TEMPO tensor network to
        time `end_time`.

        Parameters
        ----------
        progress_type: str (default = None)
            The progress report type during the computation. Types are:
            {``silent``, ``simple``, ``bar``}. If `None` then
            the default progress type is used.
        """
        if self._backend_instance.step is None:
            self._backend_instance.initialize()

        progress = get_progress(progress_type)
        with progress(self._backend_instance.num_steps) as prog_bar:
            while self._backend_instance.compute_step():
                prog_bar.update(self._backend_instance.step)
            prog_bar.update(self._backend_instance.step)

    def get_process_tensor(
            self,
            progress_type: Optional[Text] = None,
            backend: Optional[Text] = None,
            backend_config: Optional[Dict] = None) -> ProcessTensor:
        """
        Returns a the computed process tensor. It performs the computation if
        it hasn't been already done.

        Parameters
        ----------
        backend: str (default = None)
            The name of the backend for the following process tensor
            computations. If `backend` is ``None`` then the default
            backend is used.
        backend_config: dict (default = None)
            The configuration of the backend. If `backend_config` is
            ``None`` then the default backend configuration is used.

        Returns
        -------
        process_tensor: ProcessTensor
            The computed process tensor.
        """

        if self._backend_instance.step is None \
            or (self._backend_instance.step < self._backend_instance.num_steps):
            self.compute(progress_type=progress_type)

        times = self._start_time \
                + self._parameters.dt * np.arange(self._num_steps+1)
        tensors = self._backend_instance.get_tensors()
        initial_tensor = None
        name = None
        description = f"computed from {self.name} pt-tempo"
        description_dict = {
            "pt_tempo_type":str(type(self)),
            "pt_tempo_name":self.name,
            "pt_tempo_description":self.description,
            "pt_tempo_description_dict":self.description_dict,
            "pt_parameters_type":str(type(self._parameters)),
            "pt_parameters_name":self._parameters.name,
            "pt_parameters_description":self._parameters.description,
            "pt_parameters_description_dict":self._parameters.description_dict,
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
            "dt":self._parameters.dt,
            "dkmax":self._parameters.dkmax,
            "epsrel":self._parameters.epsrel,
            }

        process_tensor = ProcessTensor(
            times=times,
            tensors=tensors,
            initial_tensor=initial_tensor,
            backend=backend,
            backend_config=backend_config,
            name=name,
            description=description,
            description_dict=description_dict)

        return process_tensor


def pt_tempo_compute(
        bath: Bath,
        start_time: float,
        end_time: float,
        parameters: Optional[PtTempoParameters] = None,
        tollerance: Optional[float] = PT_DEFAULT_TOLLERANCE,
        backend: Optional[Text] = None,
        backend_config: Optional[Dict] = None,
        progress_type: Optional[Text] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None,
        description_dict: Optional[Dict] = None) -> ProcessTensor:
    """
    Shortcut for creating a process tensor by performing a PT-TEMPO
    computation.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the sytem).
    start_time: float
        The start time.
    end_time: float
        The time to which the PT-TEMPO should be computed.
    parameters: PtTempoParameters
        The parameters for the PT-TEMPO computation.
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
        parameters = guess_pt_tempo_parameters(bath=bath,
                                               start_time=start_time,
                                               end_time=end_time,
                                               tollerance=tollerance)
    ptt = PtTempo(bath,
                  start_time,
                  end_time,
                  parameters,
                  backend,
                  backend_config,
                  name,
                  description,
                  description_dict)
    ptt.compute(progress_type=progress_type)
    return ptt.get_process_tensor()

def guess_pt_tempo_parameters(
        bath: Bath,
        start_time: float,
        end_time: float,
        tollerance: Optional[float] = PT_DEFAULT_TOLLERANCE
        ) -> PtTempoParameters:
    """
    Function to roughly estimate appropriate parameters for a PT-TEMPO
    computation.

    .. warning::

        No guarantie that resulting PT-TEMPO calculation converges towards the
        correct dynamics! Please refere to the TEMPO documentation and check
        convergence by varying the parameters for PT-TEMPO manually.

    Parameters
    ----------
    bath: Bath
        The bath.
    start_time: float
        The start time.
    end_time: float
        The time to which the TEMPO should be computed.
    tollerance: float
        Tollerance for the parameter estimation.

    Returns
    -------
    pt_tempo_parameters : TempoParameters
        Estimate of appropropriate tempo parameters.
    """
    param = guess_tempo_parameters(
                bath=bath,
                start_time=start_time,
                end_time=end_time,
                tollerance=tollerance)
    return PtTempoParameters(
        dt=param.dt,
        dkmax=param.dkmax,
        epsrel=param.epsrel,
        name="Roughly estimated parameters",
        description="Estimated with 'guess_pt_tempo_parameters()'",
        description_dict={"tollerance":tollerance})

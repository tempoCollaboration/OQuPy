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
(PT-TEMPO). This module is based on [Strathearn2018], [Pollock2018],
[Jorgensen2019], and [Fux2021].

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
optimal control of non-Markovian open quantum systems*, Phys. Rev. Lett. 126,
200401 (2021).
"""

from copy import copy
from typing import Dict, Optional, Text, Union

from numpy import ndarray

from oqupy.backends.pt_tempo_backend import PtTempoBackend
from oqupy.base_api import BaseAPIClass
from oqupy.bath import Bath
from oqupy.config import PT_DEFAULT_TOLERANCE, PT_TEMPO_BACKEND_CONFIG
from oqupy.operators import left_right_super
from oqupy.process_tensor import BaseProcessTensor, \
    SimpleProcessTensor, FileProcessTensor
from oqupy.tempo import guess_tempo_parameters, influence_matrix, \
    TempoParameters
from oqupy.util import get_progress

from oqupy.backends.numerical_backend import np

PT_CLASS = {"simple": SimpleProcessTensor}

class PtTempo(BaseAPIClass):
    """
    Class to facilitate a PT-TEMPO computation.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the PT-TEMPO computation.
    start_time: float
        The start time.
    unique: bool (default = False),
        Whether to use degeneracy checking. If True reduces dimension of
        bath tensors in case of degeneracies in sums ('west') and
        sums,differences ('north') of the bath coupling operator.
        See bath:north_degeneracy_map, bath:west_degeneracy_map.
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
            start_time: float,
            end_time: float,
            parameters: TempoParameters,
            unique: Optional[bool] = False,
            process_tensor_file: Optional[Union[Text, bool]] = None,
            overwrite: Optional[bool] = False,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a PtTempo object. """
        assert isinstance(bath, Bath), \
            "Argument 'bath' must be an instance of Bath."
        self._bath = bath
        self._dimension = self._bath.dimension
        self._correlations = self._bath.correlations

        super().__init__(name, description)

        try:
            tmp_start_time = float(start_time)
        except Exception as e:
            raise AssertionError("Start time must be a float.") from e
        self._start_time = tmp_start_time

        try:
            tmp_end_time = float(end_time)
        except Exception as e:
            raise AssertionError("End time must be a float.") from e
        self._end_time = tmp_end_time

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        assert isinstance(unique, bool), \
            "Argument 'unique' must be a boolean."
        self._unique = unique

        self._process_tensor = None
        if process_tensor_file or isinstance(process_tensor_file, Text):
            if isinstance(process_tensor_file, Text):
                filename = process_tensor_file
            else:
                filename = None
            self._init_file_process_tensor(filename, overwrite)
        else:
            self._init_simple_process_tensor()

        if backend_config is None:
            self._backend_config = PT_TEMPO_BACKEND_CONFIG
        else:
            self._backend_config = backend_config

        self._coupling_comm = self._bath._coupling_comm
        self._coupling_acomm = self._bath._coupling_acomm

        tmp_num_steps = int((end_time - self._start_time)/self._parameters.dt)
        assert tmp_num_steps >= 2, \
            "Parameter `end_time` must be more than two times steps " \
            + "larger than the parameter `start_time`!"
        self._num_steps = tmp_num_steps

        self._backend_instance = None
        self._init_pt_tempo_backend()

    def _init_simple_process_tensor(self):
        """ToDo. """
        unitary = self._bath.unitary_transform
        if not np.allclose(unitary, np.identity(self._dimension)):
            transform_in = left_right_super(unitary.conjugate().T,
                                            unitary).T
            transform_out = left_right_super(unitary,
                                             unitary.conjugate().T).T
        else:
            transform_in = None
            transform_out = None
        self._process_tensor = SimpleProcessTensor(
            hilbert_space_dimension=self._dimension,
            dt=self._parameters.dt,
            transform_in=transform_in,
            transform_out=transform_out,
            name=self.name,
            description=self.description)

    def _init_file_process_tensor(self, filename, overwrite):
        """ToDo. """
        unitary = self._bath.unitary_transform
        if not np.allclose(unitary, np.identity(self._dimension)):
            transform_in = left_right_super(unitary.conjugate().T,
                                            unitary).T
            transform_out = left_right_super(unitary,
                                             unitary.conjugate().T).T
        else:
            transform_in = None
            transform_out = None

        if overwrite:
            mode = "overwrite"
        else:
            mode = "write"
        self._process_tensor = FileProcessTensor(
            mode=mode,
            filename=filename,
            hilbert_space_dimension=self._dimension,
            dt=self._parameters.dt,
            transform_in=transform_in,
            transform_out=transform_out,
            name=self.name,
            description=self.description)

    def _init_pt_tempo_backend(self):
        """Create and initialize the pt-tempo backend. """
        if self._unique:
            sum_north = np.ones(np.max(self._bath.north_degeneracy_map)+1,
                                dtype=float)
            sum_west = np.ones(np.max(self._bath.west_degeneracy_map)+1,
                               dtype=float)
            degeneracy_maps = [self._bath.north_degeneracy_map,
                               self._bath.west_degeneracy_map]
        else:
            sum_north =  np.ones(self._dimension**2, dtype=float)
            sum_west = np.ones(self._dimension**2, dtype=float)
            degeneracy_maps = None
        dkmax = self._parameters.dkmax
        if dkmax is None:
            dkmax = self._num_steps
        self._backend_instance = PtTempoBackend(
                dimension=self._dimension,
                influence=self._influence,
                process_tensor=self._process_tensor,
                sum_north=sum_north,
                sum_west=sum_west,
                num_steps=self._num_steps,
                dkmax=dkmax,
                epsrel=self._parameters.epsrel,
                config=self._backend_config,
                degeneracy_maps=degeneracy_maps)

    def _influence(self, dk: int) -> ndarray:
        """Create the influence functional matrix for a time step distance
        of dk. """
        if self._unique:
            tmp_north_deg_positions = np.array([np.where( \
                self._bath.north_degeneracy_map == i)[0][0] for i in \
                    range(np.max(self._bath.north_degeneracy_map)+1)])
            tmp_west_deg_positions = np.array([np.where( \
                self._bath.west_degeneracy_map == i)[0][0] for i in \
                    range(np.max(self._bath.west_degeneracy_map)+1)])
            tmp_deg_positions = [tmp_north_deg_positions,
                                 tmp_west_deg_positions]
        else:
            tmp_deg_positions = None

        return influence_matrix(
            dk,
            parameters=self._parameters,
            correlations=self._correlations,
            coupling_acomm=self._bath.coupling_acomm,
            coupling_comm=self._bath.coupling_comm,
            deg_positions=tmp_deg_positions)

    @property
    def dimension(self) -> ndarray:
        """Hilbert space dimension. """
        return copy(self._dimension)

    def compute(self, progress_type: Optional[Text] = None) -> None:
        """
        Propagate (or continue to propagate) the TEMPO tensor network to
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
        title = "--> PT-TEMPO computation:"
        with progress(self._backend_instance.num_steps, title) as prog_bar:
            while self._backend_instance.compute_step():
                prog_bar.update(self._backend_instance.step)
            prog_bar.update(self._backend_instance.step)

    def get_process_tensor(
            self,
            progress_type: Optional[Text] = None) -> BaseProcessTensor:
        """
        Returns a the computed process tensor. It performs the computation if
        it hasn't been already done.

        Parameters
        ----------
        progress_type: str (default = None)
            The progress report type during the computation. Types are:
            {``silent``, ``simple``, ``bar``}. If `None` then
            the default progress type is used.

        Returns
        -------
        process_tensor: SimpleProcessTensor
            The computed process tensor.
        """
        if self._backend_instance.step is None or \
            self._backend_instance.step < self._backend_instance.num_steps:
            self.compute(progress_type=progress_type)

        if len(self._process_tensor) < self._backend_instance.num_steps:
            self._backend_instance.update_process_tensor()

        return self._process_tensor


def pt_tempo_compute(
        bath: Bath,
        start_time: float,
        end_time: float,
        parameters: Optional[TempoParameters] = None,
        unique: Optional[bool] = False,
        tolerance: Optional[float] = PT_DEFAULT_TOLERANCE,
        process_tensor_file: Optional[Union[Text, bool]] = None,
        overwrite: Optional[bool] = False,
        backend_config: Optional[Dict] = None,
        progress_type: Optional[Text] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None) -> BaseProcessTensor:
    """
    Shortcut for creating a process tensor by performing a PT-TEMPO
    computation.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    start_time: float
        The start time.
    end_time: float
        The time to which the PT-TEMPO should be computed.
    parameters: TempoParameters
        The parameters for the PT-TEMPO computation.
    unique: bool (default = False),
        Whether to use degeneracy checking. If True reduces dimension of
        bath tensors in case of degeneracies in sums ('west') and
        sums,differences ('north') of the bath coupling operator.
        See bath:north_degeneracy_map, bath:west_degeneracy_map.
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
                                            tolerance=tolerance)
    ptt = PtTempo(bath,
                  start_time,
                  end_time,
                  parameters,
                  unique,
                  process_tensor_file,
                  overwrite,
                  backend_config,
                  name,
                  description)
    ptt.compute(progress_type=progress_type)
    return ptt.get_process_tensor()

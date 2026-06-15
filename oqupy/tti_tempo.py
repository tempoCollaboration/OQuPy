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
Module for the time translational invariant time evolving matrix product
operator algorithm (TTI-TEMPO). This module is based on [Link2024].

**[Link2024]**
V. Link, H. Tu, and W. T. Strunz, *Open Quantum System Dynamics from Infinite
Tensor Network Contraction*, `Phys. Rev. Lett. 132, 200403
<https://doi.org/10.1103/PhysRevLett.132.200403>`__ (2024).
"""

from typing import Dict, Optional, Text, Union

import numpy as np

from oqupy.base_api import BaseAPIClass
from oqupy.bath import Bath
from oqupy.config import TTI_TEMPO_BACKEND_CONFIG
from oqupy.process_tensor import BaseProcessTensor, SimpleProcessTensorInfinite
from oqupy.process_tensor import FileProcessTensor
from oqupy.backends.tti_tempo_backend import TTITempoBackend
from oqupy.tempo import TempoParameters
from oqupy.tempo import influence_matrix
from oqupy.operators import left_right_super
from oqupy.util import get_progress


class TTITempo(BaseAPIClass):
    """
    Class to facilitate a TTI-TEMPO computations.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    parameters: TempoParameters
        The parameters for the PT-TEMPO computation.
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
            start_time: float,
            parameters: TempoParameters,
            process_tensor_file: Optional[Union[Text, bool]] = None,
            overwrite: Optional[bool] = False,
            backend_config: Optional[Dict] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a TTITempo object. """
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

        assert isinstance(parameters, TempoParameters), \
            "Argument 'parameters' must be an instance of TempoParameters."
        self._parameters = parameters

        self._process_tensor = None
        if process_tensor_file or isinstance(process_tensor_file, Text):
            if isinstance(process_tensor_file, Text):
                filename = process_tensor_file
            else:
                filename = None
            self._init_file_process_tensor(filename, overwrite)
        else:
            self._init_infinite_process_tensor()

        if backend_config is None:
            self._backend_config = TTI_TEMPO_BACKEND_CONFIG
        else:
            self._backend_config = TTI_TEMPO_BACKEND_CONFIG | backend_config

        self._coupling_comm = np.pad(self._bath._coupling_comm, [(0, 1)])
        self._coupling_acomm = np.pad(self._bath._coupling_acomm, [(0, 1)])

        self._backend_instance = None
        self._init_tti_tempo_backend()

    def _init_infinite_process_tensor(self):
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

        self._process_tensor = SimpleProcessTensorInfinite(
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

    def _init_tti_tempo_backend(self):
        """Create and initialize the tti-tempo backend."""
        self._backend_instance = TTITempoBackend(
                dimension=self._dimension,
                influence=self._influence,
                process_tensor=self._process_tensor,
                dkmax=self._parameters.dkmax,
                epsrel=self._parameters.epsrel,
                rank=self._parameters.rank,
                config=self._backend_config)

    def _influence(self, dk):
        return influence_matrix(
            dk,
            parameters=self._parameters,
            correlations=self._correlations,
            coupling_acomm=self._coupling_acomm,
            coupling_comm=self._coupling_comm)

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
        title = "--> TTI-TEMPO computation:"
        with progress(self._backend_instance.num_steps, title) as prog_bar:
            while self._backend_instance.compute_step():
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
        process_tensor: SimpleProcessTensorInfinite
            The computed process tensor.
        """
        if self._backend_instance.step is None or \
            self._backend_instance.step < self._backend_instance.num_steps:
            self.compute(progress_type=progress_type)

        if len(self._process_tensor) < sum(
                self._backend_instance.repeating_cell):
            self._backend_instance.update_process_tensor()

        return self._process_tensor


def tti_tempo_compute(
        bath: Bath,
        start_time: float,
        parameters: TempoParameters = None,
        progress_type: Optional[Text] = None,
        process_tensor_file: Optional[Union[Text, bool]] = None,
        overwrite: Optional[bool] = False,
        backend_config: Optional[Dict] = None,
        name: Optional[Text] = None,
        description: Optional[Text] = None) -> BaseProcessTensor:
    """
    Shortcut for creating a process tensor by performing a TTI-TEMPO
    computation.

    Parameters
    ----------
    bath: Bath
        The Bath (includes the coupling operator to the system).
    start_time: float
        The start time.
    parameters: TempoParameters
        The parameters for the TTI-TEMPO computation.
    progress_type: str (default = None)
        The progress report type during the computation. Types are:
        {``'silent'``, ``'simple'``, ``'bar'``}.  If `None` then
        the default progress type is used.
    name: str (default = None)
        An optional name for the tempo object.
    description: str (default = None)
        An optional description of the tempo object.
    """
    ptt = TTITempo(bath,
                  start_time,
                  parameters,
                  process_tensor_file,
                  overwrite,
                  backend_config,
                  name,
                  description)
    ptt.compute(progress_type=progress_type)
    return ptt.get_process_tensor()

# Copyright 2022 The TEMPO Collaboration
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
Module for the process tensor approach to time evolving block decimation.
This module is based on [Fux2022].

**[Fux2022]**
G. E. Fux, D. Kilda, B. W. Lovett, and J. Keeling, *Thermalization of a
spin chain strongly coupled to its environment*, arXiv:2201.05529 (2022).

"""

from typing import Dict, List, Optional, Text, Union

import numpy as np

from oqupy.backends.pt_tebd_backend import PtTebdBackend
from oqupy.base_api import BaseAPIClass
from oqupy.config import PT_TEBD_DEFAULT_ORDER
from oqupy.config import PT_TEBD_DEFAULT_EPSREL
from oqupy.control import ChainControl
from oqupy.dynamics import Dynamics
from oqupy.mps_mpo import GateLayer, SiteGate
from oqupy.mps_mpo import compute_tebd_propagator
from oqupy.mps_mpo import AugmentedMPS
from oqupy.process_tensor import BaseProcessTensor
from oqupy.process_tensor import TrivialProcessTensor
from oqupy.system import SystemChain
from oqupy.util import get_progress

NoneType = type(None)

class PtTebdParameters(BaseAPIClass):
    r"""
    Parameters for the process tensor time evolving block decimation
    computation.

    Parameters
    ----------
    dt: float
        Length of a time step :math:`\delta t`. - It should be small enough
        such that a trotterisation between the system Hamiltonian and the
        environment it valid, and the environment auto-correlation function
        is reasonably well sampled.
    order: int
        Time evoling block decimation (TEBD) Trotterization order.
    epsrel: float
        The maximal relative error in the singular value truncation (done
        in the underlying TEBD tensor network algorithm). - It must be small
        enough such that the numerical compression does not truncate relevant
        correlations.
    name: str (default = None)
        An optional name for the PT-TEBD parameters object.
    description: str (default = None)
        An optional description of the PT-TEBD parameters object.
    """
    def __init__(
            self,
            dt: float,
            epsrel: Optional[float] = PT_TEBD_DEFAULT_EPSREL,
            order: Optional[int] = PT_TEBD_DEFAULT_ORDER,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a PtTebdParameters object."""
        self.dt = dt
        self.order = order
        self.epsrel = epsrel
        super().__init__(name, description)

    def __str__(self) -> Text:
        ret = []
        ret.append(super().__str__())
        ret.append("  dt            = {} \n".format(self.dt))
        ret.append("  order         = {} \n".format(self.order))
        ret.append("  epsrel        = {} \n".format(self.epsrel))
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
    def order(self) -> float:
        """Time evoling block decimation (TEBD) Trotterization order. """
        return self._order

    @order.setter
    def order(self, new_order: int) -> None:
        tmp_order = int(new_order)
        assert tmp_order > 0, \
            "Argument 'order' must be an integer bigger than 0."
        self._order = tmp_order

    @order.deleter
    def order(self) -> None:
        self._order = PT_TEBD_DEFAULT_ORDER

    @property
    def epsrel(self) -> float:
        """The maximal relative error in the singular value truncation. """
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

    @epsrel.deleter
    def epsrel(self) -> None:
        self._epsrel = PT_TEBD_DEFAULT_EPSREL

class PtTebd(BaseAPIClass):
    """
    Process tensor time evolving block decimation (PT-TEBD).

    Parameters
    ----------
    initial_augmented_mps: AugmentedMPS
        Initial augmented MPS.
    system_chain: SystemChain
        Object encoding the system chain Liouvillians.
    process_tensors: List[BaseProcessTensor]
        List of process tensors, one for each site.
        If a process tensor is 'None' it is assumed to be a trivial process
        tensor.
    parameters: PtTebdParameters
        PT-TEBD computation parameters.
    chain_control: ChainControl
        Optional control operations.
    start_time: float
        Optional starting time stamp.
    start_step: int
        Optional starting time step
    dynamics_sites: List[Union[int,tuple]]
        Optional list of single sites or multiple site dynamics to be recorded.
    backend_config: dict
        Optional backend configuration dictionary.

    Backend configuration `backend_config` may have the following options:

    * 'parallel' : 'multiprocess' / 'multithread'

    """
    def __init__(
            self,
            initial_augmented_mps: AugmentedMPS,
            system_chain: SystemChain,
            process_tensors: List[Union[BaseProcessTensor, NoneType]],
            parameters: PtTebdParameters,
            chain_control: Optional[ChainControl] = None,
            start_time: Optional[float] = 0.0,
            start_step: Optional[int] = 0,
            dynamics_sites: Optional[List[Union[int, tuple]]] = None,
            backend_config: Optional[Dict] = None) -> None:
        """Create a AugmentedMPS object. """

        assert isinstance(initial_augmented_mps, AugmentedMPS)
        self._initial_augmented_mps = initial_augmented_mps
        assert isinstance(system_chain, SystemChain)
        self._system_chain = system_chain

        assert isinstance(process_tensors, list)
        self._process_tensors = []
        for process_tensor in process_tensors:
            if process_tensor is None:
                self._process_tensors.append(TrivialProcessTensor())
            else:
                assert isinstance(process_tensor, BaseProcessTensor)
                self._process_tensors.append(process_tensor)

        assert isinstance(parameters, PtTebdParameters)
        self._parameters = parameters

        if chain_control is None:
            self._chain_control = ChainControl(
                hilbert_space_dimensions=self._system_chain.hs_dims)
        else:
            assert isinstance(chain_control, ChainControl)
            self._chain_control = chain_control

        assert isinstance(start_time, float)
        self._start_time = start_time

        assert isinstance(start_step, int)
        self._start_step = start_step

        if backend_config is None:
            self._backend_config = {}
        else:
            assert isinstance(backend_config, dict)
            self._backend_config = backend_config

        if dynamics_sites is not None:
            assert isinstance(dynamics_sites, list)
            tmp_dynamics_sites = []
            for sites in dynamics_sites:
                if isinstance(sites, int):
                    tmp_dynamics_sites.append(sites)
                elif isinstance(sites, tuple):
                    tmp_dynamics_sites.append(sites)
            self._dynamics_sites = tmp_dynamics_sites
        else:
            self._dynamics_sites = []

        self._tebd_propagator = None
        self._t_mps = None
        self._results = None
        self._step = None

    def initialize(self) -> None:
        """Initialize propagator, the PT-TEBD backend and results. """
        self._step = self._start_step
        self._tebd_propagator = compute_tebd_propagator(
                system_chain=self._system_chain,
                time_step=self._parameters.dt/2.0,
                epsrel=self._parameters.epsrel,
                order=self._parameters.order)
        self._results = {}
        self._t_mps = PtTebdBackend(
                gammas=self._initial_augmented_mps.gammas,
                lambdas=self._initial_augmented_mps.lambdas,
                epsrel=self._parameters.epsrel,
                config=self._backend_config)
        self._init_results()
        self._apply_controls(step=self.step, post=False)
        self._append_results()

    def _init_results(self) -> None:
        """Initialise the results dictionary. """
        pt_bond_dimensions = {}
        for site, pt in enumerate(self._process_tensors):
            if pt is not None:
                pt_bond_dimensions[site] = pt.get_bond_dimensions()

        self._results = {
            'time':[],
            'norm': [],
            'bond_dimensions': [],
            'dynamics': {},
            'pt_bond_dimensions': pt_bond_dimensions,
        }
        for sites in self._dynamics_sites:
            self._results['dynamics'][sites] = Dynamics(name=f"site{sites}")

    def _append_results(self) -> None:
        """Append new results to the results dictionionary. """
        self._t_mps.compute_traces(self._step, self._process_tensors)
        time = self.time(self._step)
        norm = self._t_mps.get_norm()
        bond_dimensions = self._t_mps.get_bond_dimensions()
        self._results['time'].append(time)
        self._results['norm'].append(norm)
        self._results['bond_dimensions'].append(bond_dimensions)
        for sites, dynamics in self._results['dynamics'].items():
            if isinstance(sites, int):
                sites_list = [sites]
            else:
                sites_list = list(sites)
            dynamics.add(
                time,
                self._t_mps.get_density_matrix(sites_list))
        self._t_mps.clear_traces()

    def _apply_controls(
            self,
            step: int,
            post: bool) -> None:
        """Apply the control operations. """
        controls = self._chain_control.get_single_site_controls(step, post)
        if controls is None:
            return

        control_gates = []
        for site, control in enumerate(controls):
            if control is not None:
                control_gates.append(SiteGate(site, control))
        control_gate_layer = GateLayer(parallel=True, gates=control_gates)
        self._t_mps.apply_site_gate_layer(control_gate_layer)

    @property
    def step(self) -> int:
        """The current step in the PT-TEBD computation. """
        return self._step

    def time(self, step: int) -> float:
        """Return the time stamp for the time step 'step'. """
        return self._start_time + self._parameters.dt*(step - self._start_step)

    @property
    def chain_control(self) -> ChainControl:
        """The chain control object. """
        return self._chain_control

    @chain_control.setter
    def chain_control(self, chain_control: ChainControl) -> None:
        if chain_control is None:
            del self.chain_control
        else:
            assert isinstance(chain_control, ChainControl)
            self._chain_control = chain_control

    @chain_control.deleter
    def chain_control(self) -> None:
        hs_dims = self._system_chain.hs_dims
        self._chain_control = ChainControl(hilbert_space_dimensions=hs_dims)

    def get_augmented_mps(self) -> AugmentedMPS:
        """Return the current AugmentedMPS. """
        gammas = []
        for i in range(len(self._t_mps)):
            gammas.append(self._t_mps.get_gamma(i))
        lambdas = []
        for i in range(len(self._t_mps) - 1):
            lambdas.append(self._t_mps.get_lambda(i))

        return AugmentedMPS(gammas, lambdas)

    def get_results(self) -> Dict:
        """Return the computed PT-TEBD results. """
        results = {}
        results['time'] = np.array(self._results['time'])
        results['norm'] = np.array(self._results['norm'])
        results['bond_dimensions'] = np.array(self._results['bond_dimensions'])
        results['dynamics'] = self._results['dynamics']
        results['pt_bond_dimensions'] = self._results['pt_bond_dimensions']
        return results

    def get_current_density_matrix(self, sites: Union[int, tuple]):
        """
        Get the current density matrix of the site(s) 'sites'.

        Parameters
        ----------
        sites: Union[int, tuple]
            The site(s).
        """
        if isinstance(sites, int):
            sites_list = [sites]
        else:
            sites_list = list(sites)
        self._t_mps.compute_traces(self._step, self._process_tensors)
        return self._t_mps.get_density_matrix(sites_list)

    def compute(
            self,
            end_step: int,
            progress_type: Text = None) -> Dict:
        """
        Perform the PT-TEBD propagation up to time step 'end_step'.

        Parameters
        ----------
        end_step: int
            The time step to which the propagation should be carried out.
        progress_type: Text
            The progress report type during the computation. Types are:
            {``'silent'``, ``'simple'``, ``'bar'``}. If `None` then
            the default progress type is used.
        """
        try:
            tmp_end_step = int(end_step)
        except Exception as e:
            raise AssertionError("End step must be an integer.") from e

        if self.step is None:
            self.initialize()

        start_step = self.step
        num_step = max(0, end_step - start_step)

        progress = get_progress(progress_type)
        with progress(num_step) as prog_bar:
            while self.step < tmp_end_step:
                self.compute_step()
                prog_bar.update(self.step - start_step)
            prog_bar.update(self.step - start_step)

        return self.get_results()

    def compute_step(self):
        """Take a step in the PT-TEBD tensor network computation. """
        self._apply_controls(step=self.step, post=True)
        self._step += 1
        for gate_layer in self._tebd_propagator.gate_layers:
            self._t_mps.apply_nn_gate_layer(gate_layer)
        self._t_mps.apply_process_tensors(self.step,
                                          self._process_tensors)
        for gate_layer in self._tebd_propagator.gate_layers:
            self._t_mps.apply_nn_gate_layer(gate_layer)
        self._apply_controls(step=self.step, post=False)
        self._append_results()

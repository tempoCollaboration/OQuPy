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
ToDo
"""

from typing import Callable, List, Optional, Text, Tuple, Union
from copy import deepcopy

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass
from oqupy.config import NpDtype

class Control(BaseAPIClass):
    """
    ToDo
    """
    def __init__(
            self,
            dimension: int,
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Creates a Control object. """
        self._dimension = dimension
        self._step_controls = {'pre':{}, 'post':{}}
        self._time_controls = {'pre':{}, 'post':{}}
        self._control_times = {'pre':np.array([]), 'post':np.array([])}
        super().__init__(name, description)

    @property
    def dimension(self):
        """Hilbert space dimension of the controlled system. """
        return self._dimension

    def add_single(
            self,
            time: Union[int, float],
            control_operation: ndarray,
            post: Optional[bool] = False) -> None:
        """
        ToDo
        """
        if post:
            pre_post = 'post'
        else:
            pre_post = 'pre'

        if isinstance(time, int):
            steps = self._step_controls[pre_post].keys()
            if time in steps:
                self._step_controls[pre_post][time] = \
                    control_operation @ self._step_controls[pre_post][time]
            else:
                self._step_controls[pre_post][time] = control_operation
        elif isinstance(time, float):
            if time in self._control_times[pre_post]:
                self._time_controls[pre_post][time] = \
                    control_operation @ self._time_controls[pre_post][time]
            else:
                self._time_controls[pre_post][time] = control_operation
                times = np.append(self._control_times[pre_post], time)
                times.sort()
                self._control_times[pre_post] = times
        else:
            raise TypeError("Parameter `time` must be either int or float.")


    def add_continuous(
            self,
            control_fct: Callable[[ndarray, float], ndarray],
            pre: Optional[bool] = True) -> None:
        """
        ToDo
        """
        raise NotImplementedError()

    def get_controls(
            self,
            step: int,
            dt: Optional[float] = None,
            start_time: Optional[float] = 0.0,
            ) -> Tuple[ndarray, ndarray]:
        """
        ToDo
        """
        pre_control_bool = False
        post_control_bool = False
        pre_control = np.identity(self.dimension**2)
        post_control = np.identity(self.dimension**2)

        # -- pre time-stamp controls --
        a = np.round((self._control_times['pre'] - start_time) / dt)
        times = np.array(self._control_times['pre'])[np.nonzero(a==step)]
        if len(times) > 0:
            print(times)
            pre_control_bool = True
            pre_control = self._time_controls['pre'][times[0]] @ pre_control
            for t in times[1:]:
                pre_control = self._time_controls['pre'][t] @ pre_control

        # -- pre step controls --
        steps = self._step_controls['pre'].keys()
        if step in steps:
            pre_control_bool = True
            pre_control = self._step_controls['pre'][step] @ pre_control

        # -- post step controls --
        steps = self._step_controls['post'].keys()
        if step in steps:
            post_control_bool = True
            post_control = self._step_controls['post'][step] @ post_control

        # -- post time-stamp controls --
        a = np.round((self._control_times['post'] - start_time) / dt)
        times = np.array(self._control_times['post'])[np.nonzero(a==step)]
        if len(times) > 0:
            post_control_bool = True
            post_control = self._time_controls['post'][times[0]] @ post_control
            for t in times[1:]:
                post_control = self._time_controls['post'][t] @ post_control

        if not pre_control_bool:
            pre_control = None
        if not post_control_bool:
            post_control = None
        return pre_control, post_control

class ChainControl(BaseAPIClass):
    """
    Control operations on a linear system chain.

    Parameters
    ----------
    hilbert_space_dimensions: List[int]
        Hilbert space dimension for each chain site.
    name: str
        An optional name for the chain controls.
    description: str
        An optional description of the chain controls.
    """
    def __init__(
            self,
            hilbert_space_dimensions: List[int],
            name: Optional[Text] = None,
            description: Optional[Text] = None) -> None:
        """Create a ChainControl object. """
        tmp_hs_dims = np.array(hilbert_space_dimensions, int)
        assert len(tmp_hs_dims.shape) == 1
        assert len(tmp_hs_dims) >= 1
        assert np.all(tmp_hs_dims > 0)
        self._hs_dims = tmp_hs_dims

        self._single_site_controls_pre = []
        self._single_site_controls_post = []

        super().__init__(name, description)

    def __len__(self):
        """Length of the chain. """
        return len(self._hs_dims)

    @property
    def hs_dims(self):
        """Hilbert space dimensions. """
        return self._hs_dims

    def add_single_site_control(
            self,
            control: ndarray,
            site: int,
            step: int,
            pre: Optional[bool] = True,
            name: Optional[Text] = None):
        """
        Add a control operation at site `site` and time step `step`.

        Parameters
        ----------
        control: ndarray
            Control operation in Liouville space.
        site: int
            Site index.
        step: int
            Timestep to which the control should be applied.
        pre: bool
            True if the control should be applied before the measurement of this
            time step.
        name: Text
            An optional name to recognize a control operation.
        """
        assert isinstance(site, int)
        assert site < len(self)
        assert isinstance(step, int)
        contr = np.array(control,  dtype=NpDtype)
        assert contr.shape == (self._hs_dims[site]**2, self._hs_dims[site]**2)
        if pre:
            self._single_site_controls_pre.append({
                "contr":contr,
                "site":site,
                "step":step,
                "name":name})
        else:
            self._single_site_controls_pre.append({
                "contr":contr,
                "site":site,
                "step":step,
                "name":name})

    def get_single_site_controls(
            self,
            step: int,
            pre: bool):
        """Get a list of single site controls for the time step `step`. """
        empty = True
        controls = [None] * len(self)

        if pre:
            ss_controls = self._single_site_controls_pre
        else:
            ss_controls = self._single_site_controls_post

        for ssc in ss_controls:
            if ssc["step"] == step:
                empty = False
                if controls[ssc["site"]] is None:
                    controls[ssc["site"]] = ssc["contr"]
                else:
                    controls[ssc["site"]] = \
                        controls[ssc["site"]] @ ssc["contr"]

        if empty:
            return None
        return deepcopy(controls)

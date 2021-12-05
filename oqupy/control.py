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

from typing import Callable, Dict, Optional, Text, Tuple, Union

import numpy as np
from numpy import ndarray

from oqupy.base_api import BaseAPIClass


class Control(BaseAPIClass):
    """
    ToDo
    """
    def __init__(
            self,
            dimension: int,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            description_dict: Optional[Dict] = None) -> None:
        """Creates a Control object. """
        self._dimension = dimension
        self._step_controls = {'pre':{}, 'post':{}}
        self._time_controls = {'pre':{}, 'post':{}}
        self._control_times = {'pre':np.array([]), 'post':np.array([])}
        super().__init__(name, description, description_dict)

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

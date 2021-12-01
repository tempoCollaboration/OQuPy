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

from typing import Callable, Optional, Tuple, Union

from numpy import ndarray


class Control:
    """
    ToDo
    """
    def __init__(self, dimension: int) -> None:
        """
        ToDo
        """
        pass # ToDo

    def add_single(
            self,
            control_operation: ndarray,
            time: Union[int, float],
            pre: Optional[bool] = True) -> None:
        """
        ToDo
        """
        pass# ToDo

    def add_continuous(
            self,
            control_fct: Callable[[ndarray, float], ndarray],
            pre: Optional[bool] = True) -> None:
        """
        ToDo
        """
        pass# ToDo

    def get_controls(
            self,
            step: int,
            dt: Optional[float] = None,
            start_time: Optional[float] = 0.0,
            ) -> Tuple[ndarray, ndarray]:
        """
        ToDo
        """
        pass # ToDo
        return None, None

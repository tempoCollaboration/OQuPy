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
Util module for tensor network backend.
"""

from typing import Any, List, Optional
from copy import copy

from numpy import zeros

def create_delta(
        tensor: Any,
        index_scrambling: List[int]) -> Any:
    """
    ToDo:
    Creates deltas in tensor.

    .. warning::
        This is a computationally inefficient method to perform the task.

    .. todo::
        Make it better.

    """
    tensor_shape = tensor.shape
    a = [0]*len(tensor_shape)

    ret_shape = tuple(list(tensor_shape)[i] for i in index_scrambling)
    ret_ndarray = zeros(ret_shape, dtype=tensor.dtype)

    # emulating a do-while loop
    do_while_condition = True
    while do_while_condition:
        tensor_indices = tuple(a)
        ret_indices = tuple(a[i] for i in index_scrambling)
        ret_ndarray[ret_indices] = tensor[tensor_indices]
        do_while_condition = increase_list_of_index(a, tensor_shape)

    return ret_ndarray

def add_singleton(
        tensor: Any,
        index: int = -1) -> Any:
    """
    ToDo
    """
    ret = copy(tensor)
    shape_list = list(ret.shape)
    shape_list.insert(index, 1)
    ret.shape = tuple(shape_list)
    return ret

def increase_list_of_index(
        a: List,
        shape: List,
        index: Optional[int] = -1) -> bool:
    """
    ToDo
    """
    a[index] += 1
    if a[index] >= shape[index]:
        if index == -len(shape):
            return False
        a[index] = 0
        return increase_list_of_index(a, shape, index-1)
    return True

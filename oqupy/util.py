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
Module for utilities.
"""

import sys
import copy as cp
from typing import List, Optional, Text
from threading import Timer
from time import time
from datetime import timedelta

import numpy as np
from numpy import ndarray

from oqupy.config import PROGRESS_TYPE

# -- numpy utils --------------------------------------------------------------

def create_delta(
        tensor: ndarray,
        index_scrambling: List[int]) -> ndarray:
    """
    Creates deltas in numpy tensor.

    .. warning::
        This is a computationally inefficient method to perform the task.

    .. todo::
        Make it better.

    """
    tensor_shape = tensor.shape
    a = [0]*len(tensor_shape)

    ret_shape = tuple(list(tensor_shape)[i] for i in index_scrambling)
    ret_ndarray = np.zeros(ret_shape, dtype=tensor.dtype)

    # emulating a do-while loop
    do_while_condition = True
    while do_while_condition:
        tensor_indices = tuple(a)
        ret_indices = tuple(a[i] for i in index_scrambling)
        ret_ndarray[ret_indices] = tensor[tensor_indices]
        do_while_condition = increase_list_of_index(a, tensor_shape)

    return ret_ndarray

def increase_list_of_index(
        a: List,
        shape: List,
        index: Optional[int] = -1) -> bool:
    """Circle through a list of indices. """
    a[index] += 1
    if a[index] >= shape[index]:
        if index == -len(shape):
            return False
        a[index] = 0
        return increase_list_of_index(a, shape, index-1)
    return True

def add_singleton(
        tensor: ndarray,
        index: Optional[int] = -1,
        copy: Optional[bool] = True) -> ndarray:
    """Add a singleton to a numpy tensor. """
    if copy:
        ten = cp.copy(tensor)
    else:
        ten = tensor
    shape_list = list(ten.shape)
    shape_list.insert(index, 1)
    ten.shape = tuple(shape_list)
    return ten

def is_diagonal_matrix(tensor: ndarray):
    """Check if matrix is diagonal """
    assert len(tensor.shape) == 2
    i, j = tensor.shape
    assert i == j
    test = tensor.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

# -- process bar --------------------------------------------------------------

class BaseProgress:
    """Base class to display computation progress. """

    def __enter__(self):
        """Contextmanager enter. """
        raise NotImplementedError()

    def __exit__(self, exception_type, exception_value, traceback):
        """Contextmanager exit. """
        raise NotImplementedError()

    def update(self, step=None):
        """Update the progress. """
        raise NotImplementedError()


class ProgressSilent(BaseProgress):
    """Class NOT to display the computation progress. """
    def __init__(self, max_value):
        """Create a ProgressSilent object. """
        self.max_value = max_value
        self.step = None

    def __enter__(self):
        """Contextmanager enter. """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Contextmanager exit. """
        pass

    def update(self, step=None):
        """Update the progress. """
        self.step = step


class ProgressSimple(BaseProgress):
    """Class to display the computation progress step by step. """
    def __init__(self, max_value):
        """Create a ProgressSimple object. """
        self.max_value = max_value
        self.step = None
        self._start_time = None
        self._previouse_time = None

    def __enter__(self):
        """Contextmanager enter. """
        self._start_time = time()
        self._previouse_time = time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Contextmanager exit. """
        current_time = time()
        total_t = current_time - self._start_time
        print("Total elapsed time:  {:9.1f}s".format(total_t), flush=True)

    def update(self, step=None):
        """Update the progress. """
        current_time = time()
        dt = current_time - self._previouse_time
        total_t = current_time - self._start_time
        self.step = step
        print("Step {:5d} of {:5d},  total time: {:9.1f}s (+{:8.2f}s)".format(
            self.step, self.max_value, total_t, dt), flush=True)
        self._previouse_time = current_time


PROGRESS_BAR_LENGTH = 40

class ProgressBar(BaseProgress):
    """Class to display the computation progress with a nice progress bar. """
    def __init__(self, max_value):
        """Create a ProgressBar object. """
        self._timer = None
        self._start_time = time()
        self._file = sys.stdout
        self._max_value = max_value
        self._length = PROGRESS_BAR_LENGTH
        self._step = None

    def __enter__(self):
        """Contextmanager enter. """
        self._timer = Timer(1.0, self._print_status)
        self._timer.start()
        return self

    def _print_status(self):
        if self._step is None:
            step = 0
        else:
            step = self._step
        try:
            frac = float(step)/float(self._max_value)
        except ZeroDivisionError:
            frac = 1.0
        delta_t = time() - self._start_time
        time_string = "{:0>8}".format(str(timedelta(seconds=int(delta_t))))
        done_int = int(frac*self._length)
        bar_string = "\r{:5.1f}% {:4d} of {:4d} [{}{}] {}"
        bar_string = bar_string.format(frac*100,
                                       step,
                                       self._max_value,
                                       "#" * done_int,
                                       "-" * (self._length - done_int),
                                       time_string)
        self._file.write(bar_string)
        self._file.flush()

    def __exit__(self, exception_type, exception_value, traceback):
        """Contextmanager exit. """
        self._timer.cancel()
        self._print_status()
        delta_t = time() - self._start_time
        print("\nElapsed time: {:.1f}s".format(delta_t),
              file=self._file,
              flush=True)

    def update(self, step=None):
        """Update the progress. """
        self._timer.cancel()
        self._timer = Timer(1.0, self.update)
        self._timer.start()
        if step is not None:
            self._step = step
        self._print_status()


PROGRESS_DICT = {
    "silent": ProgressSilent,
    "simple": ProgressSimple,
    "bar": ProgressBar,
    }

def get_progress(progress_type: Text = None) -> BaseProgress:
    """Get a progress class from the progress_type. """
    if progress_type is None:
        progress_type = PROGRESS_TYPE
    assert progress_type in PROGRESS_DICT, \
        "Unknown progress_type='{}', know are {}".format(
            progress_type, PROGRESS_DICT.keys())
    return PROGRESS_DICT[progress_type]

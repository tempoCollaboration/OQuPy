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

import copy as cp
from datetime import timedelta
import sys
from threading import Timer
from time import time
from typing import Any, List, Optional, Text

from numpy import ndarray

from oqupy.config import PROGRESS_TYPE

from oqupy.backends.numerical_backend import np

# -- numpy utils --------------------------------------------------------------

def create_deltas(
        func_tensors: callable,
        indices: List[int],
        index_scrambling: List[int],
        scale: float=1.0) -> List[ndarray]:
    """Creates deltas in multiple tensors."""
    # use a test tensor to obtain the indices
    tensor = func_tensors(indices[0])
    _shape = np.array(tensor.shape, dtype=int)
    _idxs = np.array(index_scrambling, dtype=int)
    _indices = get_indices(_shape, np.prod(_shape))
    indices_in = tuple(_indices)
    indices_out = tuple(_indices[_idxs])

    # accumulate scrambled tensors and return list
    # TODO: vectorize and make it JIT-compatibile
    scrambled_tensors = []
    for i in indices:
        scrambled_tensors.append(np.update(
            array=np.zeros(tuple(_shape[_idxs]), \
                            dtype=tensor.dtype),
            indices=indices_out,
            values=func_tensors(i)[indices_in] / scale
        ))
    return scrambled_tensors

def create_delta(
        tensor: ndarray,
        index_scrambling: List[int],
        scale: float=1.0) -> ndarray:
    """Creates deltas in a tensor."""
    # converting to NumPy-array for future-proof implementation
    # see [this issue](https://github.com/google/jax/issues/4564)
    # the shape of the tensor has n_in elements whereas
    # index_scrambling has n_out elements
    _shape = np.array(tensor.shape, dtype=int)
    _idxs = np.array(index_scrambling, dtype=int)

    # obtain the indices from the `get_indices` function
    _indices = get_indices(_shape, np.prod(_shape))
    indices_in = tuple(_indices)
    indices_out = tuple(_indices[_idxs])

    # scramble output tensor with elements of input tensor
    return np.update(
        array=np.zeros(tuple(_shape[_idxs]), \
                        dtype=tensor.dtype),
        indices=indices_out,
        values=tensor[indices_in] / scale
    )

def get_indices(shape: ndarray,
                n_iters: int) -> ndarray:
    """Obtain index matrix for scrambling."""
    # obtain divisors for each axis with shape equal to the
    # number of elements contained in preceding axes
    # for example, a tensor with shape (4, 5, 3) will result in
    # [15, 3, 1] to divide each axes and obtain the remainder
    # modullo the dimension of each axis of the tensor
    divisors = np.cumprod(np.concatenate([
        shape[1:],
        np.array([1], dtype=int)
    ])[::-1])[::-1]
    # prepare an iteration matrix of shape (n_in, n_iters)
    # to index each element, for e.g., n_iters = 3 x 5 x 4
    iteration_matrix = np.arange(0, n_iters).reshape(
        (n_iters, 1)).repeat(shape.shape[0], 1)
    # obtain the indices for each axes of input and output tensors
    return ((iteration_matrix / divisors).astype(int) % shape).T

def create_delta_old(
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
        ret_ndarray = np.update(
            array=ret_ndarray,
            indices=ret_indices,
            values=tensor[tensor_indices]
        )
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

# -- input parsing -----------------------------------------------------------

def check_convert(
        variable: Any,
        conv_type: Any,
        name: Text = None,
        msg: Text = None):
    """Attempt to convert variable into a specific type. """
    try:
        converted_variable = conv_type(variable)
    except Exception as e:
        name_str = f"`{name}`" if name is not None else ""
        msg_str = msg if msg is not None else ""
        err_str = f"Variable `{name_str}` must be type `{conv_type.__name__}`."
        raise TypeError(err_str + msg_str) from e
    return converted_variable

def check_true(
        expr: bool,
        msg: Text = None):
    """Check that an specific expression is true. """
    if not expr:
        msg_str = msg if msg is not None else ""
        raise ValueError(msg_str)

def check_isinstance(
        variable: Any,
        types: Any,
        name: Text = None,
        msg: Text = None):
    """Check that a variable is an instance of one of the given types. """
    if not isinstance(types, tuple):
        types_list = (types, )
    else:
        types_list = types
    if not isinstance(variable, types_list):
        name_str = f"`{name}`" if name is not None else ""
        types_str = " or ".join([f"`{type.__name__}`" for type in types_list])
        msg_str = msg if msg is not None else ""
        raise TypeError(f"Variable {name_str} is not of the type "
                        + f"{types_str}. {msg_str}")

# -- process bar --------------------------------------------------------------

class BaseProgress:
    """Base class to display computation progress. """

    def __enter__(self):
        """Contextmanager enter. """
        return self.enter()

    def __exit__(self, exception_type, exception_value, traceback):
        """Contextmanager exit. """
        self.exit()

    def enter(self):
        """Context enter."""
        raise NotImplementedError()

    def exit(self):
        """Context exit. """
        raise NotImplementedError()

    def update(self, step=None):
        """Update the progress. """
        raise NotImplementedError()


class ProgressSilent(BaseProgress):
    """Class NOT to display the computation progress. """
    def __init__(self, max_value, title = None):
        """Create a ProgressSilent object. """
        self.max_value = max_value
        self.title = title
        self.step = None

    def enter(self):
        """Context enter. """
        return self

    def exit(self):
        """Context exit. """
        pass

    def update(self, step=None):
        """Update the progress. """
        self.step = step


class ProgressSimple(BaseProgress):
    """Class to display the computation progress step by step. """
    def __init__(self, max_value, title = None):
        """Create a ProgressSimple object. """
        self.max_value = max_value
        self.title = title
        self.step = None
        self._file = sys.stdout
        self._start_time = None
        self._previouse_time = None

    def enter(self):
        """Context enter. """
        if self.title is not None:
            print(self.title, flush=True)
        self._start_time = time()
        self._previouse_time = time()
        return self

    def exit(self):
        """Context exit. """
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
    def __init__(self, max_value, title = None):
        """Create a ProgressBar object. """
        self._timer = None
        self._start_time = time()
        self._file = sys.stdout
        self.max_value = max_value
        self.title = title
        self._length = PROGRESS_BAR_LENGTH
        self._step = None

    def enter(self):
        """Context enter. """
        if self.title is not None:
            print(self.title, file=self._file, flush=True)
        self._timer = Timer(1.0, self._print_status)
        self._timer.start()
        return self

    def _print_status(self):
        if self._step is None:
            step = 0
        else:
            step = self._step
        try:
            frac = float(step)/float(self.max_value)
        except ZeroDivisionError:
            frac = 1.0
        delta_t = time() - self._start_time
        time_string = "{:0>8}".format(str(timedelta(seconds=int(delta_t))))
        done_int = int(frac*self._length)
        bar_string = "\r{:5.1f}% {:4d} of {:4d} [{}{}] {}"
        bar_string = bar_string.format(frac*100,
                                       step,
                                       self.max_value,
                                       "#" * done_int,
                                       "-" * (self._length - done_int),
                                       time_string)
        self._file.write(bar_string)
        self._file.flush()

    def exit(self):
        """Context exit. """
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

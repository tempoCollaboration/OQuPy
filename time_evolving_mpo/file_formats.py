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
Module to define/check the export/import file formats.
"""

from typing import Dict, Text, Union

from numpy import ndarray

from time_evolving_mpo.config import NpDtype, NpDtypeReal
from time_evolving_mpo.util import load_object


def _assert_common_fields(data_dict: Dict) -> None:
    """Assert that the fields common to all file formats exist. """
    # check that the version field exists and is of type "Text".
    assert "version" in data_dict
    assert isinstance(data_dict["version"], Text)
    # check that the name field exists and is of type "Text".
    assert "name" in data_dict
    assert isinstance(data_dict["name"], (type(None), Text))
    # check that the description field exists and is of type "Text".
    assert "description" in data_dict
    assert isinstance(data_dict["description"], (type(None), Text))
    # check that the description_dict field exists and is of type "Dict".
    assert "description_dict" in data_dict
    assert isinstance(data_dict["description_dict"], (type(None), Dict))


# --- definition of the .tempoDynamics file format ----------------------------

def assert_tempo_dynamics_dict(t_dyn_dict: Dict) -> None:
    """
    Assert that the data is of correct .tempoDynamics form, which is:

    .. code-block:: python

        Dict[ "version"          : "1.0",
              "name"             : None / Text,
              "description"      : None / Text,
              "description_dict" : None / Dict,
              "times"            : ndarray,
              "states"           : ndarray ]

    where `times` is a 1D array of floats and `states` is a 1D array of square
    matrices of the same length.

    Parameters
    ----------
    t_dyn_dict: dict
        Data dictionary of the .tempoDynamics file.

    Raises
    ------
    `AssertionError`:
        If `t_dyn_dict` is not of the correct .tempoDynamics form.
    """
    _assert_common_fields(t_dyn_dict)
    assert t_dyn_dict["version"] == "1.0"
    # check that the times field exists and is of type "ndarray".
    assert "times" in t_dyn_dict
    assert isinstance(t_dyn_dict["times"], ndarray)
    # check that the states field exists and is of type "ndarray".
    assert "states" in t_dyn_dict
    assert isinstance(t_dyn_dict["states"], ndarray)

    # check that the dtype of times is NpDtypeReal
    assert t_dyn_dict["times"].dtype == NpDtypeReal
    # check that the dtype of states is NpDtype
    assert t_dyn_dict["states"].dtype == NpDtype
    # check that length of times and states is equal.
    assert len(t_dyn_dict["times"]) == len(t_dyn_dict["states"])
    # check that the matrices are states are square matrices
    if len(t_dyn_dict["times"]) != 0:
        assert len(t_dyn_dict["states"].shape) == 3
        assert t_dyn_dict["states"].shape[1] == t_dyn_dict["states"].shape[2]

def assert_tempo_dynamics_file(filename: Text) -> None:
    """
    Assert that the file is of correct .tempoDynamics form
    [see tempo.assert_tempo_dynamics_file() for more details].

    Parameters
    ----------
    filename: str
        Path to the file.

    Raises
    ------
    `AssertionError`:
        If the data found in the file is not of the correct .tempoDynamics
        form.
    """
    t_dyn_dict = load_object(filename)
    assert_tempo_dynamics_dict(t_dyn_dict)

def check_tempo_dynamics_file(filename: Text) -> bool:
    """
    Check if file is of correct .tempoDynamics form.
    [see tempo.assert_tempo_dynamics_file() for more details].

    Parameters
    ----------
    filename: str
        Path to the file.

    Returns
    -------
    True or False:
    """
    try:
        assert_tempo_dynamics_file(filename)
        return True
    except AssertionError:
        return False

def print_tempo_dynamics_dict(t_dyn_dict: Dict) -> None:
    """Print header of .processTensor data. """
    assert_tempo_dynamics_dict(t_dyn_dict)
    print("Tempo Dynamics:")
    for key, value in t_dyn_dict.items():
        if key not in ["description_dict",
                       "times",
                       "states"]:
            print("  {:20s} = {}".format(key, value))
        if key == "times":
            value = "steps:{},  min:{},  max:{}".format(len(value),
                                                        min(value),
                                                        max(value))
            print("  {:20s} = {}".format(key, value))
        if key == "states":
            value = "shape:{}".format(value.shape)
            print("  {:20s} = {}".format(key, value))
        if key == "description_dict":
            print("  {:20s} = {}".format(key, "dict:"))
            if value is not None:
                for key_des, value_des in value.items():
                    print("  {:20s}     {:15s} : {}".format(" ",
                                                            key_des,
                                                            value_des))

def print_tempo_dynamics_file(filename: Text) -> None:
    """Print header of .processTensor file. """
    t_dyn_dict = load_object(filename)
    print_tempo_dynamics_dict(t_dyn_dict)

# --- definition of the .processTensor file format ----------------------------

def _assert_process_tensor_tensors(
        initial_tensor: Union[None, ndarray],
        tensors: list) -> None:
    """Assert that the tensors of the process tensor have the right shapes. """

    assert isinstance(initial_tensor, (type(None), ndarray))

    if isinstance(initial_tensor, ndarray):
        assert initial_tensor.dtype == NpDtype
        assert len(initial_tensor.shape) == 2

    assert isinstance(tensors, list)
    for tensor in tensors:
        assert isinstance(tensor, ndarray)
        assert tensor.dtype == NpDtype
        assert len(tensor.shape) == 3 \
            or len(tensor.shape) == 4

    # ---- if len(tensors) == 0: --------------

    if len(tensors) == 0:
        if initial_tensor is None:
            pass
        else:
            assert initial_tensor.shape[0] == 1
        return

    # ---- if len(tensors) != 0: --------------

    # dimension of physical legs
    dim = tensors[0].shape[2]

    if initial_tensor is None:
        assert tensors[0].shape[0] == 1
    else:
        assert initial_tensor.shape[0] == tensors[0].shape[0]
        assert initial_tensor.shape[1] == dim

    for tensor in tensors:
        assert tensor.shape[2] == dim
        if len(tensor.shape) == 4:
            assert tensor.shape[3] == dim

    for i in range(len(tensors)-1):
        assert tensors[i].shape[1] == tensors[i+1].shape[0]

    assert tensors[-1].shape[1] == 1


def assert_process_tensor_dict(p_t_dict: Dict) -> None:
    """
    Assert that the data is of correct .processTensor form, which is:

    .. code-block:: python

        Dict[ "version"          : "1.0",
              "name"             : None / Text,
              "description"      : None / Text,
              "description_dict" : None / Dict,
              "times"            : None / float / ndarray,
              "initial_tensor"   : None / ndarray,
              "tensors"          : List[ndarray] ]

    If the field `times` is `None` this amounts to no information on
    the time slots of the process tensor (only "initial step", "first step",
    etc). If the field `times` is a `float` it signals that the time steps are
    uniformly spaced. If the field `times` is an `numpy.ndarray` it
    has to be a vector of the time slots considered in this process tensor in
    ascending order. In this case the length of `times` must be the length of
    `tensors` plus 1.

    If the field `initial_tensor` is `None` this amounts to no given initial
    state. If the field `initial_tensor` is an `numpy.ndarray` it must be a
    2-legged tensor (i.e. a matrix) where the first leg is the internal leg
    connecting to the next part of the array of tensors that represent the
    process tensor. The second leg is vectorised initial state (in fact the
    first slot in the process tensor).

    The field `tensors` is list of three or four legged tensors. The first and
    second legs are the internal legs that connect to the previous and next
    tensor. If `initial_tensor` is `None` the first leg of the first tensor
    must be a dummy leg of dimension 1. The  second leg of the last tensor must
    always be a dummy leg of dimension 1. The third leg is the "incoming" leg
    of the previous time slot, while the fourth leg is the "resulting" leg of
    the following time slot. If the tensor has only three legs, a
    Kronecker-delta between the third and fourth leg is assumed.

    Parameters
    ----------
    p_t_dict: dict
        Data dictionary of the process tensor.

    Raises
    ------
    `AssertionError`:
        If `p_t_dict` is not of the correct process tensor form.
    """
    _assert_common_fields(p_t_dict)
    assert p_t_dict["version"] == "1.0"
    # check that the field times exists and is of one of the three types.
    assert "times" in p_t_dict
    assert isinstance(p_t_dict["times"], (type(None), float, ndarray))
    # check that the field initial_tensors exists and is None or an ndarray.
    assert "initial_tensor" in p_t_dict
    assert isinstance(p_t_dict["initial_tensor"], (type(None), ndarray))
    # check that the field tensors exists and is a list.
    assert "tensors" in p_t_dict
    assert isinstance(p_t_dict["tensors"], list)

    if isinstance(p_t_dict["times"], ndarray):
        assert len(p_t_dict["times"]) == len(p_t_dict["tensors"])+1


    _assert_process_tensor_tensors(p_t_dict["initial_tensor"],
                                   p_t_dict["tensors"])

def assert_process_tensor_file(filename: Text) -> None:
    """
    Assert that the file is of correct .processTensor form
    [see tempo.assert_process_tensor_file() for more details].

    Parameters
    ----------
    filename: str
        Path to the file.

    Raises
    ------
    `AssertionError`:
        If the data found in the file is not of the correct .processTensor
        form.
    """
    t_dyn_dict = load_object(filename)
    assert_process_tensor_dict(t_dyn_dict)

def check_process_tensor_file(filename: Text) -> bool:
    """
    Check if file is of correct .processTensor form
    [see tempo.assert_process_tensor_file() for more details].

    Parameters
    ----------
    filename: str
        Path to the file.

    Returns
    -------
    True or False:
    """
    try:
        assert_process_tensor_file(filename)
        return True
    except AssertionError:
        return False

def print_process_tensor_dict(p_t_dict: Dict) -> None:
    """Print header of .processTensor data. """
    assert_process_tensor_dict(p_t_dict)
    print("Process Tensor:")
    for key, value in p_t_dict.items():
        if key not in ["description_dict",
                       "times",
                       "initial_tensor",
                       "tensors"]:
            print("  {:20s} = {}".format(key, value))
        if key == "description_dict":
            print("  {:20s} = {}".format(key, "dict:"))
            if value is not None:
                for key_des, value_des in value.items():
                    print("  {:20s}     {:15s} : {}".format(" ",
                                                            key_des,
                                                            value_des))

def print_process_tensor_file(filename: Text) -> None:
    """Print header of .processTensor file. """
    p_t_dict = load_object(filename)
    print_process_tensor_dict(p_t_dict)

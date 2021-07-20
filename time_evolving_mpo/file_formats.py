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
Module to define/check the export/import file formats.
"""

from typing import Dict, Text

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
    True or False: bool
        True if the file has the correct format.
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

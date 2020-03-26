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
Module for utillities.
"""

import pickle
from typing import Any, Text

from numpy import identity, kron, ndarray


# -- superoperators ----------------------------------------------------------

def commutator(operator: ndarray) -> ndarray:
    """Construct commutator superoperator from operator."""
    dim = operator.shape[0]
    return kron(operator, identity(dim)) - kron(identity(dim), operator.T)

def acommutator(operator: ndarray) -> ndarray:
    """Construct anti-commutator superoperator from operator."""
    dim = operator.shape[0]
    return kron(operator, identity(dim)) + kron(identity(dim), operator.T)

def left_super(operator: ndarray) -> ndarray:
    """Construct left acting superoperator from operator."""
    dim = operator.shape[0]
    return kron(operator, identity(dim))

def right_super(operator: ndarray) -> ndarray:
    """Construct right acting superoperator from operator."""
    dim = operator.shape[0]
    return kron(identity(dim), operator.T)

def left_right_super(
        left_operator: ndarray,
        right_operator: ndarray) -> ndarray:
    """Construct left and right acting superoperator from operators."""
    return kron(left_operator, right_operator.T)


# -- save and load from file --------------------------------------------------

def save_object(obj: Any, filename: Text, overwrite: bool) -> None:
    """
    Save an object to a file using pickle.

    """
    if overwrite:
        mode = 'wb'
    else:
        mode = 'xb'
    with open(filename, mode) as file:
        pickle.dump(obj, file)

def load_object(filename: Text) -> Any:
    """
    ToDo
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

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
Tests for the time_evovling_mpo.util module.
"""

import pytest

import numpy as np

from time_evolving_mpo.util import commutator, acommutator
from time_evolving_mpo.util import left_super, right_super, left_right_super
from time_evolving_mpo.util import save_object, load_object


# -- testing super operators --------------------------------------------------

N = 3

a = np.random.rand(N,N) + 1j * np.random.rand(N,N)
b = np.random.rand(N,N) + 1j * np.random.rand(N,N)
x = np.random.rand(N,N) + 1j * np.random.rand(N,N)

a_dagger = a.conjugate().T
b_dagger = a.conjugate().T
x_dagger = a.conjugate().T

x_vector = x.flatten()
x_dagger_vector = x_dagger.flatten()


def test_commutator():
    sol = a@x - x@a
    res = commutator(a)@x_vector
    np.testing.assert_almost_equal(res.reshape(N,N),sol)

def test_acommutator():
    sol = a@x + x@a
    res = acommutator(a)@x_vector
    np.testing.assert_almost_equal(res.reshape(N,N),sol)

def test_left_super():
    sol = a@x
    res = left_super(a)@x_vector
    np.testing.assert_almost_equal(res.reshape(N,N),sol)

def test_right_super():
    sol = x@a
    res = right_super(a)@x_vector
    np.testing.assert_almost_equal(res.reshape(N,N),sol)

def test_left_right_super():
    sol = a@x@b
    res = left_right_super(a,b)@x_vector
    np.testing.assert_almost_equal(res.reshape(N,N),sol)


some_obj_A = ["hi","there!"]
some_obj_B = 3

def test_save_object():
    filename = "tests/data/temp.saveObjectTest"
    save_object(some_obj_A, filename, overwrite=True)
    with pytest.raises(FileExistsError):
        save_object(some_obj_B, filename, overwrite=False)
    obj = load_object(filename)
    assert obj[0] == "hi"

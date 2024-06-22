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
Tests for the oqupy.backends.node_array module.

.. todo::
    This is quite a mess. Make it better!
"""
import pytest

import numpy as np
import tensornetwork as tn

import oqupy.backends.node_array as node_array


def test_node_array():
    np.random.seed(0)
    a = np.random.rand(3,4,9,5)
    b = np.random.rand(5,5,9,6)
    c = np.random.rand(6,5,9,2)
    d = np.random.rand(2,3,9)
    d2 = np.random.rand(2,3,9,3)
    na = node_array.NodeArray([a,b,c,d], right=False, name="my node array")
    na2 = node_array.NodeArray([a,b,c,d2], name="my node array")
    str(na)
    str(na2)
    na.bond_dimensions
    na.rank
    del na.name
    na.get_verbose_string()

    dd1 = node_array.NodeArray([d], name="D1", left=False, right=False)
    dd2 = node_array.NodeArray([d], name="D2", left=False, right=True)
    dd3 = node_array.NodeArray([d], name="D3", left=True, right=False)
    dd4 = node_array.NodeArray([d], name="D4", left=True, right=True)

    del dd1.name
    dd2.get_verbose_string()

    #Apply matrix (left/right)
    m = np.random.rand(3,7)
    na.apply_matrix(m,left=True)
    na2.apply_matrix(m,left=False)

    # Apply vector (left/right)
    v = np.random.rand(7)
    na.apply_vector(v, left=True)
    na2.apply_vector(v, left=False)

    na = node_array.NodeArray([])
    assert na.rank == None

def test_node_array_join_split():
    np.random.seed(0)
    a = np.random.rand(3,4,9,5)
    b = np.random.rand(5,5,9,6)
    c = np.random.rand(6,5,9,2)
    d = np.random.rand(2,3,9)
    x = np.random.rand(2,9,3)

    na0 = node_array.NodeArray([x], left=False, name="array 0")
    na1 = node_array.NodeArray([a,b], name="array 1")
    na2 = node_array.NodeArray([c,d], right=False, name="array 2")

    na_12 = node_array.join(na1,na2)
    na_012 = node_array.join(na0, na_12)
    na_A, na_B = node_array.split(na_012, 2)

    with pytest.raises(IndexError):
        na_A, na_B = node_array.split(na_012, 5)

def test_node_array_svd():
    np.random.seed(0)
    a = np.random.rand(3,4,9,5)
    b = np.random.rand(5,5,9,6)
    c = np.random.rand(6,5,9,2)
    d = np.random.rand(2,3,9)
    na = node_array.NodeArray([a,b,c,d], right=False, name="my node array")

    na.svd_sweep(1, 3, max_singular_values=4)
    na.svd_sweep(-1, 0, max_singular_values=1)
    na.svd_sweep(0, -1, max_singular_values=1)

    with pytest.raises(IndexError):
        na.svd_sweep(5, 3, max_singular_values=4)

    with pytest.raises(IndexError):
        na.svd_sweep(1, 5, max_singular_values=4)

def test_node_array_contractions():
    np.random.seed(0)
    mps = node_array.NodeArray([np.random.rand(3,2,3),
                                np.random.rand(3,3,3),
                                np.random.rand(3,4,3)],
                               name="MPS")
    mpo1 = node_array.NodeArray([np.random.rand(2,5,4),
                                 np.random.rand(4,3,5,4),
                                 np.random.rand(4,4,5)],
                                left=False,
                                right=False,
                                name="MPO1")
    mpo2 = node_array.NodeArray([np.random.rand(5,5,3),
                                 np.random.rand(3,5,5,3),
                                 np.random.rand(3,5,5)],
                                left=False,
                                right=False,
                                name="MPO2")
    mps.zip_up(mpo1, [(0, 0)], left_index=0, right_index=-1, max_singular_values=10)

    mps.zip_up(mpo2, left_index=0, max_singular_values=11)

    mps.svd_sweep(-1,0,max_singular_values=2);

    # ### 2nd example

    mps = node_array.NodeArray([np.random.rand(4,3),
                                np.random.rand(3,4,3),
                                np.random.rand(3,4)],
                               left=False,
                               right=False,
                               name="MPS")
    mps1 = mps.copy()
    mps2 = mps.copy()
    mps3 = mps.copy()
    mps4 = mps.copy()
    mps5 = mps.copy()
    mpsTwo = node_array.NodeArray([np.random.rand(4,3),
                                 np.random.rand(3,4)],
                                left=False,
                                right=False,
                                name="MPS")
    mpsL = node_array.NodeArray([np.random.rand(3,4,3),
                                 np.random.rand(3,4)],
                                left=True,
                                right=False,
                                name="MPS")
    mpsR = node_array.NodeArray([np.random.rand(4,3),
                                 np.random.rand(3,4,3)],
                                left=False,
                                right=True,
                                name="MPS")
    mpo1 = node_array.NodeArray([np.random.rand(3,4,4,3),
                                 np.random.rand(3,4,4)],
                                left=True,
                                right=False,
                                name="MPO1")
    mpo5 = mpo1.copy()
    mpo2 = node_array.NodeArray([np.random.rand(4,4,3),
                                 np.random.rand(3,4,4,3)],
                                left=False,
                                right=True,
                                name="MPO2")
    arr1 = node_array.NodeArray([np.random.rand(4,2,2,3),
                                 np.random.rand(3,4,2,2,3),
                                 np.random.rand(3,4,2,2,3)],
                                left=False,
                                right=True,
                                name="array1")
    arr2 = node_array.NodeArray([np.random.rand(3,2,4,2,3),
                                 np.random.rand(3,2,4,2,3),
                                 np.random.rand(3,2,4,2)],
                                left=True,
                                right=False,
                                name="array2")

    mps1a = mps1.copy()
    mps1b = mps1.copy()
    mps1c = mps1.copy()
    with pytest.raises(ValueError):
        mps1a.zip_up(mpo1, [(0,0)], left_index=0, direction="blub")

    with pytest.raises(IndexError):
        mps1b.zip_up(mpo1, [(0,0)], left_index=6, direction="left")

    with pytest.raises(IndexError):
        mps1c.zip_up(mpo1, [(0,0)], right_index=-6, direction="left")

    mps1.zip_up(mpo1, [(0, 0)], left_index=0, direction="left")

    mps2.zip_up(mpo2, [(0, 0)], left_index=1, right_index=2, direction="left")

    mps3.zip_up(arr1, [(0, 0)])

    mps3.zip_up(arr2, [(0, 0),(1, 2)], right_index=-1, direction="left", max_singular_values=7)

    mps4b = mps4.copy()
    mps4c = mps4.copy()
    mps4d = mps4.copy()
    mps4.contract(mps, [(0,0)], direction="right")

    with pytest.raises(ValueError):
        mps4b.contract(mps, [(0,0)], direction="blub")
    with pytest.raises(AssertionError):
        mps4c.contract(mpsTwo, [(0,0)], right_index=-1, direction="right")
    with pytest.raises(AssertionError):
        mps4d.contract(mpsTwo, [(0,0)], left_index=0, direction="left")

    mps1.contract(mps)
    mps1.nodes

    mps5b = mps5.copy()
    mps5.contract(mpsL, [(0,0)], left_index=0, direction="right")
    mps5b.contract(mpsR, [(0,0)], right_index=-1, direction="left")

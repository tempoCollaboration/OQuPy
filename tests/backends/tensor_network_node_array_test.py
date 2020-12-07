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
Tests for the time_evolving_mpo.backends.node_array module.
"""
import sys

# sys.path.insert(0,".")

import pytest
import numpy as np
import tensornetwork as tn
import time_evolving_mpo.backends.tensor_network.node_array as node_array


def test_node_array():
    np.random.seed(0)
    a = np.random.rand(3,4,9,5)
    b = np.random.rand(5,5,9,6)
    c = np.random.rand(6,5,9,2)
    d = np.random.rand(2,3,9)
    na = node_array.NodeArray([a,b,c,d], right=False, name="my node array")
    print(na.bond_dimensions)
    print(na.rank)

    dd = node_array.NodeArray([d], name="D1", left=False, right=False)
    print(dd)
    print(f"    rank = {dd.rank}\n")
    dd = node_array.NodeArray([d], name="D2", left=False, right=True)
    print(dd)
    print(f"    rank = {dd.rank}\n")
    dd = node_array.NodeArray([d], name="D3", left=True, right=False)
    print(dd)
    print(f"    rank = {dd.rank}\n")
    dd = node_array.NodeArray([d], name="D4", left=True, right=True)
    print(dd)
    print(f"    rank = {dd.rank}\n")

    #Apply matrix (left/right)
    m = np.random.rand(3,7)
    print(na)
    na.apply_matrix(m,left=True)
    print(na)

    # Apply vector (left/right)
    v = np.random.rand(7)
    print(na)
    na.apply_vector(v, left=True)
    print(na)


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
    print(na0)
    print(na1)
    print(na2)


    na_12 = node_array.join(na1,na2)
    print(na_12)


    na_012 = node_array.join(na0, na_12)
    print(na_012)

    # ## Split

    na_A, na_B = node_array.split(na_012, 2)


    print(na_A)
    print(na_B)


def test_node_array_svd():
    np.random.seed(0)
    a = np.random.rand(3,4,9,5)
    b = np.random.rand(5,5,9,6)
    c = np.random.rand(6,5,9,2)
    d = np.random.rand(2,3,9)
    na = node_array.NodeArray([a,b,c,d], right=False, name="my node array")

    na.svd_sweep(1, 3, max_singular_values=4)
    print(na)

    singular_values = na.svd_sweep(-1, 0, max_singular_values=1)
    print(na)
    print("Singular values:")
    for keep, discard in singular_values:
        print(f"  [keep / discard]: {keep} / {discard}")

def test_node_array_contractions():
    np.random.seed(0)
    mps = node_array.NodeArray([np.random.rand(3,2,3),
                                np.random.rand(3,3,3),
                                np.random.rand(3,4,3)],
                               name="MPS")
    print(mps)
    print(f"    rank:{mps.rank}")
    mpo1 = node_array.NodeArray([np.random.rand(2,5,4),
                                 np.random.rand(4,3,5,4),
                                 np.random.rand(4,4,5)],
                                left=False,
                                right=False,
                                name="MPO1")
    print(mpo1)
    print(f"    rank:{mpo1.rank}")
    mpo2 = node_array.NodeArray([np.random.rand(5,5,3),
                                 np.random.rand(3,5,5,3),
                                 np.random.rand(3,5,5)],
                                left=False,
                                right=False,
                                name="MPO2")
    print(mpo2)
    print(f"    rank:{mpo2.rank}")


    mps.zip_up(mpo1, [(0, 0)], left_index=0, right_index=-1, max_singular_values=10)
    print(mps)


    mps.zip_up(mpo2, [(0, 0)], left_index=0, max_singular_values=11)
    print(mps)


    mps.svd_sweep(-1,0,max_singular_values=2);
    print(mps)

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
    mpsL = node_array.NodeArray([np.random.rand(3,4,3),
                                 np.random.rand(3,4)],
                                left=True,
                                right=False,
                                name="MPS")
    print(mps)
    print(f"    rank:{mps.rank}")
    print(mpsL)
    print(f"    rank:{mpsL.rank}")
    mpo1 = node_array.NodeArray([np.random.rand(3,4,4,3),
                                 np.random.rand(3,4,4)],
                                left=True,
                                right=False,
                                name="MPO1")
    mpo5 = mpo1.copy()
    print(mpo1)
    print(f"    rank:{mpo1.rank}")
    mpo2 = node_array.NodeArray([np.random.rand(4,4,3),
                                 np.random.rand(3,4,4,3)],
                                left=False,
                                right=True,
                                name="MPO2")
    print(mpo2)
    print(f"    rank:{mpo2.rank}")
    arr1 = node_array.NodeArray([np.random.rand(4,2,2,3),
                                 np.random.rand(3,4,2,2,3),
                                 np.random.rand(3,4,2,2,3)],
                                left=False,
                                right=True,
                                name="array1")
    print(arr1)
    print(f"    rank:{arr1.rank}")
    arr2 = node_array.NodeArray([np.random.rand(3,2,4,2,3),
                                 np.random.rand(3,2,4,2,3),
                                 np.random.rand(3,2,4,2)],
                                left=True,
                                right=False,
                                name="array2")
    print(arr2)
    print(f"    rank:{arr2.rank}")


    mps1.zip_up(mpo1, [(0, 0)], left_index=0, direction="left")
    print(mps1)


    mps2.zip_up(mpo2, [(0, 0)], left_index=1, right_index=2, direction="left")
    print(mps2)


    mps3.zip_up(arr1, [(0, 0)])
    print(mps3)


    mps3.zip_up(arr2, [(0, 0),(1, 2)], right_index=-1, direction="left", max_singular_values=7)
    print(mps3)


    mps4.contract(mps, [(0,0)], direction="right")
    print(mps4)
    print(mps4.nodes)


    print(mps1)
    print(mps)
    mps1.contract(mps, [(0,0)])
    print(mps1)
    mps1.nodes


    # mps5.contract(mpsL, [(0,0)], left_index=0, direction="left")


    mps5.contract(mpsL, [(0,0)], left_index=0, direction="right")
    print(mps5)

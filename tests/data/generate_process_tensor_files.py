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
Skript to generate data files.
"""

import sys
import pickle

import numpy as np

# -----------------------------------------------------------------------------

def save_object(obj, filename, overwrite=False) -> None:
    """Save an object to a file using pickle. """
    if overwrite:
        mode = 'wb'
    else:
        mode = 'xb'
    with open(filename, mode) as file:
        pickle.dump(obj, file)

def c_rand(*args):
    return np.random.rand(*args) + 1.0j*np.random.rand(*args)

# -----------------------------------------------------------------------------

def bad_file_A():
    filename = "test_v1_0_bad_file_A.processTensor"
    p_t_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":"bla", # times must be one of: None / float / ndarray
                "initial_tensor":None,
                "tensors":[]}
    save_object(p_t_dict, filename, True)

def good_file_A():
    filename = "test_v1_0_good_file_A.processTensor"
    p_t_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":None,
                "initial_tensor":None,
                "tensors":[]}
    save_object(p_t_dict, filename, True)

def good_file_B():
    filename = "test_v1_0_good_file_B.processTensor"
    np.random.seed(1)
    dim = 5
    N1 = 7
    N2 = 11
    N3 = 13
    tensors = [c_rand(1,N1,dim,dim),
               c_rand(N1,N2,dim),
               c_rand(N2,N3,dim,dim),
               c_rand(N3,1,dim)]
    p_t_dict = {"version":"1.0",
                "name":"Alex the little",
                "description":"barely 5 foot 1",
                "description_dict":None,
                "times":0.05,
                "initial_tensor":None,
                "tensors":tensors}
    save_object(p_t_dict, filename, True)

def good_file_C():
    filename = "test_v1_0_good_file_C.processTensor"
    np.random.seed(1)
    dim = 5
    N1 = 7
    N2 = 11
    N3 = 13
    times = np.array([-0.5,0.0,0.6,2.0,4.0])
    initial_tensor = c_rand(N3,dim)
    tensors = [c_rand(N3,N1,dim,dim),
               c_rand(N1,N2,dim),
               c_rand(N2,N3,dim,dim),
               c_rand(N3,1,dim)]
    p_t_dict = {"version":"1.0",
                "name":"Alexander the great",
                "description":"also barely 5 foot 1",
                "description_dict":None,
                "times":times,
                "initial_tensor":initial_tensor,
                "tensors":tensors}
    save_object(p_t_dict, filename, True)

def good_file_D():
    filename = "test_v1_0_good_file_D.processTensor"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0])
    initial_tensor = c_rand(1,dim)
    p_t_dict = {"version":"1.0",
                "name":"Eve",
                "description":"The initial state",
                "description_dict":{"alpha":0.3, "epsilon":3.3e-7},
                "times":times,
                "initial_tensor":initial_tensor,
                "tensors":[]}
    save_object(p_t_dict, filename, True)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    bad_file_A()
    good_file_A()
    good_file_B()
    good_file_C()
    good_file_D()

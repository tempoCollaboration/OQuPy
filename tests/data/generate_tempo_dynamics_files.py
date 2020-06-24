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
    filename = "test_v1_0_bad_file_A.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0],dtype="float64")
    states = np.array([c_rand(dim,dim+1)]) # state(s) must be square matrices
    t_d_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)

def bad_file_B():
    filename = "test_v1_0_bad_file_B.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0],dtype="float64")
    states = np.array([c_rand(dim,dim),
                       c_rand(dim,dim)]) # wrong because two states
                                         #       but only one time
    t_d_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)

def bad_file_C():
    filename = "test_v1_0_bad_file_C.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0, 0.1, 0.2, 0.4], dtype="float64")
    states = c_rand(4,dim,dim)
    name = "Ralph"
    description = "wonderful"
    description_dict = {"alpha":0.3, "omega":4.0}
    t_d_dict = {"version":"1.?", # must be exactly "1.0"
                "name":name,
                "description":description,
                "description_dict":description_dict,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)


def good_file_A():
    filename = "test_v1_0_good_file_A.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0],dtype="float64")
    states = np.array([c_rand(dim,dim)])
    t_d_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)

def good_file_B():
    filename = "test_v1_0_good_file_B.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0, 1.0],dtype="float64")
    states = np.array([c_rand(dim,dim),
                       c_rand(dim,dim)])
    t_d_dict = {"version":"1.0",
                "name":None,
                "description":None,
                "description_dict":None,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)

def good_file_C():
    filename = "test_v1_0_good_file_C.tempoDynamics"
    np.random.seed(1)
    dim = 5
    times = np.array([0.0, 0.1, 0.2, 0.4], dtype="float64")
    states = c_rand(4,dim,dim)
    name = "Ralph"
    description = "wonderful"
    description_dict = {"alpha":0.3, "omega":4.0}
    t_d_dict = {"version":"1.0",
                "name":name,
                "description":description,
                "description_dict":description_dict,
                "times":times,
                "states":states}
    save_object(t_d_dict, filename, True)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    bad_file_A()
    bad_file_B()
    bad_file_C()
    good_file_A()
    good_file_B()
    good_file_C()

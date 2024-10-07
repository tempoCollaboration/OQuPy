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
Tests for the time_evovling_mpo.operators module.
"""

from oqupy import operators

from oqupy.backends.numerical_backend import default_np, np

SIGMA = {"id":[[1, 0], [0, 1]],
         "x":[[0, 1], [1, 0]],
         "y":[[0, -1j], [1j, 0]],
         "z":[[1, 0], [0, -1]],
         "+":[[0, 1], [0, 0]],
         "-":[[0, 0], [1, 0]]}

SPIN_DM = {"up":[[1, 0], [0, 0]],
           "down":[[0, 0], [0, 1]],
           "z+":[[1, 0], [0, 0]],
           "z-":[[0, 0], [0, 1]],
           "x+":[[0.5, 0.5], [0.5, 0.5]],
           "x-":[[0.5, -0.5], [-0.5, 0.5]],
           "y+":[[0.5, -0.5j], [0.5j, 0.5]],
           "y-":[[0.5, 0.5j], [-0.5j, 0.5]],
           "mixed":[[0.5, 0.0], [0.0, 0.5]]}


def test_identity():
    for n in {1,2,7}:
        result = operators.identity(n)
        default_np.testing.assert_array_almost_equal(result,np.identity(n))

def test_operators_sigma():
    for name in SIGMA:
        result = operators.sigma(name)
        default_np.testing.assert_array_equal(result,SIGMA[name])

def test_operators_spin_dm():
    for name in SPIN_DM:
        result = operators.spin_dm(name)
        default_np.testing.assert_array_equal(result,SPIN_DM[name])

def test_operators_create_destroy():
    for n in {1,2,7}:
        adag = operators.create(n)
        a = operators.destroy(n)
        result = adag@a
        default_np.testing.assert_array_almost_equal(result,np.diag(np.array(range(n), dtype=np.dtype_complex)))

# -- testing super operators --------------------------------------------------

N = 3

random_floats = np.get_random_floats(1234, (6, N, N))
a = random_floats[0] + 1j * random_floats[1]
b = random_floats[2] + 1j * random_floats[3]
x = random_floats[4] + 1j * random_floats[5]

a_dagger = a.conjugate().T
b_dagger = a.conjugate().T
x_dagger = a.conjugate().T

x_vector = x.flatten()
x_dagger_vector = x_dagger.flatten()


def test_commutator():
    sol = a@x - x@a
    res = operators.commutator(a)@x_vector
    default_np.testing.assert_array_almost_equal(res.reshape(N,N),sol)

def test_acommutator():
    sol = a@x + x@a
    res = operators.acommutator(a)@x_vector
    default_np.testing.assert_array_almost_equal(res.reshape(N,N),sol)

def test_left_super():
    sol = a@x
    res = operators.left_super(a)@x_vector
    default_np.testing.assert_array_almost_equal(res.reshape(N,N),sol)

def test_right_super():
    sol = x@a
    res = operators.right_super(a)@x_vector
    default_np.testing.assert_array_almost_equal(res.reshape(N,N),sol)

def test_left_right_super():
    sol = a@x@b
    res = operators.left_right_super(a,b)@x_vector
    default_np.testing.assert_array_almost_equal(res.reshape(N,N),sol)

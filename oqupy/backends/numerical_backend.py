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
Module containing NumPy-like and SciPy-like numerical backends.
"""

import numpy as default_np
import scipy.linalg as default_la

import oqupy.config as oc

class NumPy:
    """
    The NumPy backend employing
    dynamic switching through `oqupy.config`.
    """
    @property
    def backend(self) -> default_np:
        """Getter for the backend."""
        return oc.NUMERICAL_BACKEND_NUMPY

    @property
    def dtype_complex(self) -> default_np.dtype:
        """Getter for the complex datatype."""
        return oc.NumPyDtypeComplex

    @property
    def dtype_float(self) -> default_np.dtype:
        """Getter for the float datatype."""
        return oc.NumPyDtypeFloat

    def __getattr__(self,
                    name: str,
                    ):
        """Return the backend's default attribute."""
        backend = object.__getattribute__(self, 'backend')
        return getattr(backend, name)

    def update(self,
               array,
               indices:tuple,
               values,
               ) -> default_np.ndarray:
        """Option to update select indices of an array with given values."""
        if not isinstance(array, default_np.ndarray):
            return array.at[indices].set(values)
        array[indices] = values
        return array

    def get_random_floats(self,
                          seed,
                          shape,
                          ):
        """Method to obtain random floats with a given seed and shape."""
        backend = object.__getattribute__(self, 'backend')
        random_floats = default_np.random.default_rng(seed).random(shape, \
                        dtype=default_np.float64)
        return backend.array(random_floats, dtype=self.dtype_float)

class LinAlg:
    """
    The Linear Algebra backend employing
    dynamic switching through `oqupy.config`.
    """
    @property
    def backend(self) -> default_la:
        """Getter for the backend."""
        return oc.NUMERICAL_BACKEND_LINALG

    def __getattr__(self, name: str):
        """Return the backend's default attribute."""
        backend = object.__getattribute__(self, 'backend')
        return getattr(backend, name)

# initialize for import
np = NumPy()
la = LinAlg()

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

import os

import numpy as default_np
import scipy.linalg as default_la

from tensornetwork.backend_contextmanager import \
    set_default_backend

import oqupy.config as oc

# store instances of the initialized backends
# this way, `oqupy.config` remains unchanged
# and `ocupy.config.DEFAULT_BACKEND` is used
# when NumPy and LinAlg are initialized
NUMERICAL_BACKEND_INSTANCES = {}

def get_numerical_backends(
        backend_name: str,
    ):
    """Function to get numerical backend.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Options are `'jax'` and `'numpy'`.

    Returns
    -------
    backends: list
        NumPy and LinAlg backends.
    """

    _bn = backend_name.lower()
    if _bn in NUMERICAL_BACKEND_INSTANCES:
        set_default_backend(_bn)
        return NUMERICAL_BACKEND_INSTANCES[_bn]
    assert _bn in ['jax', 'numpy'], \
        "currently supported backends are `'jax'` and `'numpy'`"

    if 'jax' in _bn:
        try:
            # explicitly import and configure jax
            import jax
            import jax.numpy as jnp
            import jax.scipy.linalg as jla
            jax.config.update('jax_enable_x64', True)

            # # TODO: GPU memory allocation (default is 0.75)
            # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
            # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'

            # set TensorNetwork backend
            set_default_backend('jax')

            NUMERICAL_BACKEND_INSTANCES['jax'] = [jnp, jla]
            return NUMERICAL_BACKEND_INSTANCES['jax']
        except ImportError:
            print("JAX not installed, defaulting to NumPy")

    # set TensorNetwork backend
    set_default_backend('numpy')

    NUMERICAL_BACKEND_INSTANCES['numpy'] = [default_np, default_la]
    return NUMERICAL_BACKEND_INSTANCES['numpy']

class NumPy:
    """
    The NumPy backend employing
    dynamic switching through `oqupy.config`.
    """
    def __init__(self,
                 backend_name=oc.DEFAULT_BACKEND,
                 ):
        """Getter for the backend."""
        self.backend = get_numerical_backends(backend_name)[0]

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
        return getattr(self.backend, name)

    def update(self,
               array,
               indices: tuple,
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
        random_floats = default_np.random.default_rng(seed).random(shape, \
                        dtype=default_np.float64)
        return self.backend.array(random_floats, dtype=self.dtype_float)

class LinAlg:
    """
    The Linear Algebra backend employing
    dynamic switching through `oqupy.config`.
    """
    def __init__(self,
                 backend_name=oc.DEFAULT_BACKEND,
                 ):
        """Getter for the backend."""
        self.backend = get_numerical_backends(backend_name)[1]

    def __getattr__(self,
                    name: str,
                    ):
        """Return the backend's default attribute."""
        return getattr(self.backend, name)

# setup libraries using environment variable
# fall back to oqupy.config.DEFAULT_BACKEND
try:
    BACKEND_NAME = os.environ[oc.BACKEND_ENV_VAR]
except KeyError:
    BACKEND_NAME = oc.DEFAULT_BACKEND
np = NumPy(backend_name=BACKEND_NAME)
la = LinAlg(backend_name=BACKEND_NAME)

def set_numerical_backends(
        backend_name: str
    ):
    """Function to set numerical backend.

    Parameters
    ----------
    backend_name: str
        Name of the backend. Options are `'jax'` and `'numpy'`.
    """
    backends = get_numerical_backends(backend_name)
    np.backend = backends[0]
    la.backend = backends[1]

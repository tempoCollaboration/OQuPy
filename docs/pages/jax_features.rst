Experimental Support for GPUs/TPUs
==================================
The current development branch "dev/jax" implements experimental support
for GPUs/TPUs.

Although OQuPy is built on top of the backend-agnostic
`TensorNetwork <https://github.com/google/TensorNetwork>`__ library,
OQuPy uses vanilla NumPy and SciPy throughout its implementation.

The "dev/jax" branch adds supports for GPUs/TPUs via the
`JAX <https://jax.readthedocs.io/en/latest/>`__ library. A new
``oqupy.backends.numerical_backend.py`` module handles the
`breaking changes in JAX
NumPy <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`__,
while the rest of the modules utilizes ``numpy`` and ``scipy.linalg``
instances from there without explicitly importing JAX-based libraries.

Enabling Experimental Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable experimental features, switch to the ``dev/jax`` branch and use

.. code:: python

   from oqupy.backends import enable_jax_features
   enable_jax_features()

Alternatively, the `OQUPY_BACKEND` environmental variable may be set to `jax` to
initialize the jax backend by default.

Contributing Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

To contribute features compatible with the JAX backend,
please adhere to the following set of guidelines:

-  avoid wildcard imports of NumPy and SciPy.
-  use ``from oqupy.backends.numerical_backend import np`` instead of
   ``import numpy as np`` and use the alias ``default_np`` in cases
   vanilla NumPy is explicitly required.
-  use ``from oqupy.backends.numerical_backend import la`` instead of
   ``import scipy.linalg as la``, except that for non-symmetric
   eigen-decomposition, ``scipy.linalg.eig`` should be used.
-  use one of ``np.dtype_complex`` (``np.dtype_float``) or
   ``oqupy.config.NumPyDtypeComplex`` (``oqupy.config.NumPyDtypeFloat``)
   instead of ``np.complex_`` (``np.float_``).
-  convert lists or tuples to arrays when passing them as arguments
   inside functions.
-  use ``array = np.update(array, indices, values)`` instead of
   ``array[indices] = values``.
-  use ``np.get_random_floats(seed, shape)`` instead of
   ``np.random.default_rng(seed).random(shape)``.
-  declare signatures for ``np.vectorize`` explicitly.
-  avoid directly changing the ``shape`` attribute of an array (use
   ``.reshape`` instead)

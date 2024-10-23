# Development

The current "development" branch of OQuPy implements

* [Experimental Support for GPUs/TPUs](#experimental-support-for-gpustpus)

## Experimental Support for GPUs/TPUs

Although OQuPy is built on top of the backend-agnostic
[TensorNetwork](https://github.com/google/TensorNetwork) library,
OQuPy uses vanilla NumPy and SciPy throughout its implementation.

This "experimental" approach implements backend support for [JAX](https://jax.readthedocs.io/en/latest/)
through a new `oqupy.backends.numerical_backend.py` module which handles the
[breaking changes in JAX NumPy](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html),
while the rest of the modules utilizes `numpy` and `scipy.linalg` instances from there
without explicitly importing JAX-based libraries.

### Enabling Experimental Features

To enable experimental features in the , use
```python
from oqupy.backends import enable_experimental_features
enable_experimental_features()
```

### Contributing to Experimental Feautres

If you wish to contribute to these "experimental" features,
an optional set of guidelines can be followed to reduce the maintenance overhead.
These are, to:

* avoid wildcard imports of NumPy and SciPy.
* use `from oqupy.backends.numerical_backend import np` instead of `import numpy as np` and use the alias `default_np` for vanilla NumPy.
* use `from oqupy.backends.numerical_backend import la` instead of `import scipy.linalg as la`, except that for non-symmetric eigen-decomposition, `scipy.linalg.eig` should be used.
* use one of `np.dtype_complex` (`np.dtype_float`) or `oqupy.config.NumPyDtypeComplex` (`oqupy.config.NumPyDtypeFloat`) instead of `np.complex_` (`np.float_`).
* convert lists or tuples to arrays when passing them as arguments inside functions.
* use `array = np.update(array, indices, values)` instead of `array[indices] = values`.
* use `np.get_random_floats(seed, shape)` instead of `np.random.default_rng(seed).random(shape)`.
* declare signatures for `np.vectorize` explicitly.
* avoid in-place updates of `shape` attribute.

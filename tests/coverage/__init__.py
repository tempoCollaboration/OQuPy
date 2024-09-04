from importlib.util import find_spec

if find_spec('jax') is not None:
    # JAX configuration
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jla
    import oqupy.config as oc
    import tensornetwork as tn
    jax.config.update('jax_enable_x64', True)
    oc.NUMERICAL_BACKEND_NUMPY = jnp
    oc.NumPyDtypeComplex = jnp.complex128
    oc.NumPyDtypeFloat = jnp.float64
    oc.NUMERICAL_BACKEND_LINALG = jla
    tn.set_default_backend('jax')

    # # TODO: GPU memory allocation (default is 0.75)
    # import os
    # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'
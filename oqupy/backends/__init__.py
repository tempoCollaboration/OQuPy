"""Module to initialize OQuPy's backends."""

from oqupy.backends.numerical_backend import set_numerical_backends

def enable_jax_features():
    """Function to enable experimental features."""

    # set numerical backend to JAX
    set_numerical_backends('jax')

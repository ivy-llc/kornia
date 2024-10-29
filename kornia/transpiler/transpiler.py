"""Module for transpiling Kornia to other frameworks."""

import kornia
from kornia.core.external import ivy


def to_jax():
    """Convert Kornia to JAX.

    Transpiles the Kornia library to JAX using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to JAX

    Example:
        >>> import kornia
        >>> jax_kornia = kornia.to_jax()
        >>> import jax
        >>> input = jax.random.normal(jax.random.key(42), shape=(2, 3, 4, 5))
        >>> gray = jax_kornia.color.gray.rgb_to_grayscale(input)
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="jax",
    )


def to_numpy():
    """Convert Kornia to NumPy.

    Transpiles the Kornia library to NumPy using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to NumPy

    Example:
        >>> import kornia
        >>> np_kornia = kornia.to_numpy()
        >>> import numpy as np
        >>> input = np.random.normal(size=(2, 3, 4, 5))
        >>> gray = np_kornia.color.gray.rgb_to_grayscale(input)

    Note:
        Ivy does not currently support transpiling trainable modules to NumPy.
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="numpy",
    )


def to_tensorflow():
    """Convert Kornia to TensorFlow.

    Transpiles the Kornia library to TensorFlow using [ivy](https://github.com/ivy-llc/ivy). The transpilation process
    occurs lazily, so the transpilation on a given kornia function/class will only occur when it's called or
    instantiated for the first time. This will make any functions/classes slow when being used for the first time,
    but any subsequent uses should be as fast as expected.

    Return:
        The Kornia library transpiled to TensorFlow

    Example:
        >>> import kornia
        >>> tf_kornia = kornia.to_tensorflow()
        >>> import tensorflow as tf
        >>> input = tf.random.normal((2, 3, 4, 5))
        >>> gray = tf_kornia.color.gray.rgb_to_grayscale(input)
    """
    return ivy.transpile(
        kornia,
        source="torch",
        target="tensorflow",
    )


def set_backend(backend: str = "torch") -> None:
    """Converts Kornia to the chosen backend framework inplace.

    Transpiles the Kornia library to the chosen backend framework using [ivy](https://github.com/ivy-llc/ivy).
    The transpilation process occurs lazily, so the transpilation on a given kornia function/class will only
    occur when it's called or instantiated for the first time. This will make any functions/classes slow when
    being used for the first time, but any subsequent uses should be as fast as expected.

    Args:
        backend (str, optional): The backend framework to transpile Kornia to.
                                 Must be one of ["jax", "numpy", "tensorflow", "torch"].
                                 Defaults to "torch".

    Example:
        >>> import kornia
        >>> kornia.set_backend("tensorflow")
        >>> import tensorflow as tf
        >>> input = tf.random.normal((2, 3, 4, 5))
        >>> gray = kornia.color.gray.rgb_to_grayscale(input)
    """
    import sys

    kornia_module = sys.modules["kornia"]
    backend = backend.lower()

    assert backend in ["jax", "numpy", "tensorflow", "torch"], 'Backend framework must be one of "jax", "numpy", "tensorflow", or "torch"'

    ivy.transpile(
        kornia_module,
        source="torch",
        target=backend,
        inplace=True,  # TODO: add this functionality to ivy
    )

    # TODO: unwrap and re-wrap the kornia module if, say, it's already converted to jax and the user wants to convert it to tensorflow
    # TODO: ensure that torch -> torch works fine by returning the existing module
    # TODO: ensure that framework -> torch works fine by unwrapping the existing module

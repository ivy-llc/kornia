import ivy
import kornia


def to_jax(backend_compile: bool = False):
    """Convert Kornia to JAX.

    Transpiles the Kornia library to JAX using [ivy](https://ivy.dev). The transpilation process 
    occurs lazily, so the transpilation on a given kornia function will only occur when it's called 
    for the first time. This will make the initial call of any function in the transpiled library 
    slower, but subsequent calls should be as fast as expected.

    Args:
        backend_compile: whether to apply the native compiler, jax.jit, to transpiled kornia functions

    Return:
        the transpiled kornia library

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
        to="jax",
        backend_compile=backend_compile,
    )


def to_numpy():
    """Convert Kornia to NumPy.

    Transpiles the Kornia library to NumPy using [ivy](https://ivy.dev). The transpilation process 
    occurs lazily, so the transpilation on a given kornia function will only occur when it's called 
    for the first time. This will make the initial call of any function in the transpiled library 
    slower, but subsequent calls should be as fast as expected.

    Return:
        the transpiled kornia library

    Example:
        >>> import kornia
        >>> np_kornia = kornia.to_numpy()
        >>> import numpy as np
        >>> input = np.random.normal(size=(2, 3, 4, 5))
        >>> gray = np_kornia.color.gray.rgb_to_grayscale(input)
    """
    return ivy.transpile(
        kornia,
        source="torch",
        to="numpy",
    )


def to_tensorflow(backend_compile: bool = False):
    """Convert Kornia to TensorFlow.

    Transpiles the Kornia library to TensorFlow using [ivy](https://ivy.dev). The transpilation process 
    occurs lazily, so the transpilation on a given kornia function will only occur when it's called 
    for the first time. This will make the initial call of any function in the transpiled library 
    slower, but subsequent calls should be as fast as expected.

    Args:
        backend_compile: whether to apply the native compiler, tf.function, to transpiled kornia functions

    Return:
        the kornia library converted to TensorFlow

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
        to="tensorflow",
        backend_compile=backend_compile,
    )

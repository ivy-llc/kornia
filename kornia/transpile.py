import ivy

def unify(x, args=None, kwargs=None):
    """
    Alias for to_ivy, transpiles a Kornia Callable or set of them from the torch framework to the Ivy Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.unify(x, source="torch", args=args, kwargs=kwargs) 

def to_ivy(x, args=None, kwargs=None):
    """
    Transpiles a Kornia Callable or set of them from the torch framework to the Ivy Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.transpile(x, source="torch", to='ivy', args=args, kwargs=kwargs)

def to_tensorflow(x, args=None, kwargs=None):
    """
    Transpiles a Kornia Callable or set of them from the torch framework to the TensorFlow Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.transpile(x, source="torch", to='tensorflow', args=args, kwargs=kwargs)

def to_jax(x, args=None, kwargs=None):
    """
    Transpiles a Kornia Callable or set of them from the torch framework to the JAX Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.transpile(x, source="torch", to='jax', args=args, kwargs=kwargs)

def to_numpy(x, args=None, kwargs=None):
    """
    Transpiles a Kornia Callable or set of them from the torch framework to the NumPy Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.transpile(x, source="torch", to='numpy', args=args, kwargs=kwargs)

def to_paddle(x, args=None, kwargs=None):
    """
    Transpiles a Kornia Callable or set of them from the torch framework to the PaddlePaddle Framework.
    If args or kwargs are specified, transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters:
    - x: Native callable(s) to transpile.
    - args: Arguments to pass to the function.
    - kwargs: Keyword arguments to pass to the function.

    Returns:
    - A transpiled Graph or a non-initialized LazyGraph. If the object is a native trainable module,
      the corresponding module in the target framework will be returned. If the object is a ModuleType,
      the function will return a copy of the module with every method lazily transpiled.
    """
    return ivy.transpile(x, source="torch", to='paddle', args=args, kwargs=kwargs)
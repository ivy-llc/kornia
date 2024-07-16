.. raw:: html

   <div style="display: block;" align="center">
       <img class="only-dark" width="30%" src="https://raw.githubusercontent.com/ivy-llc/ivy-llc.github.io/main/src/assets/full_logo_dark_long.png#gh-dark-mode-only"/>
   </div>

.. raw:: html

   <div style="display: block;" align="center">
       <img class="only-light" width="30%" src="https://raw.githubusercontent.com/ivy-llc/ivy-llc.github.io/main/src/assets/full_logo_light_long.png#gh-light-mode-only"/>
   </div>

Multi-Framework Support
=======================

Kornia can now be used with `Numpy <https://numpy.org/>`_, `TensorFlow <https://www.tensorflow.org/>`_, 
and `JAX <https://jax.readthedocs.io/en/latest/index.html>`_ 
thanks to an integration with `Ivy <https://github.com/ivy-llc/ivy>`_. 

This can be accomplished using the following functions, which are now part of the kornia api:

* :code:`kornia.to_tensorflow`

* :code:`kornia.to_jax`

* :code:`kornia.to_numpy`

Here's an example of using kornia with TensorFlow:

.. code:: python

    import kornia
    import tensorflow as tf

    tf_kornia = kornia.to_tensorflow()

    rgb_image = tf.random.normal((1, 3, 224, 224))
    gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)

So what's happening here? Let's break it down.

#. Transpiling kornia to TensorFlow

    This line lazily transpiles every function in kornia to TensorFlow. Because the transpilation happens lazily, no function will be
    transpiled until it is actually called.

    .. code-block:: python

        tf_kornia = kornia.to_tensorflow()

#. Calling a TF kornia function

    We can now call any kornia function with TF arguments. However, this function will be very slow relative to the original function - 
    as the function is being transpiled during this step.

    .. code-block:: python

        rgb_image = tf.random.normal((1, 3, 224, 224))
        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # slow

#. Subsequent function calls

    The good news is any calls of the function after the initial call will be much faster, as it has already been transpiled, 
    and should approximately match the speed of the torch function.

    .. code-block:: python

        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # fast

#. Backend compilation

    The transpiled functions can be made even faster by using a backend compiler (`tf.function` or `jax.jit`).
    This can be done simply by setting `backend_compile=True`. However, it should be noted that not all transpiled
    functions are compatible with backend compilation, and backend compilation is not available with NumPy.

    .. code-block:: python

        import kornia
        import tensorflow as tf

        tf_kornia = kornia.to_tensorflow(backend_compile=True)

        rgb_image = tf.random.normal((1, 3, 224, 224))
        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # slow
        gray_image = tf_kornia.color.rgb_to_grayscale(rgb_image)  # fast


Kornia can be used with JAX and NumPy in the same way:

.. code:: python

    import kornia
    import numpy as np

    np_kornia = kornia.to_numpy()

    rgb_image = np.random.normal(size=(1, 3, 224, 224))
    gray_image = np_kornia.color.rgb_to_grayscale(rgb_image)


.. code:: python

    import kornia
    import jax

    jax_kornia = kornia.to_jax()

    rgb_image = jax.random.normal(jax.random.key(42), shape=(1, 3, 224, 224))
    gray_image = jax_kornia.color.rgb_to_grayscale(rgb_image)


Limitations
-----------

The primary limitation of ivy's transpiler in its current form is that it uses a function tracing approach, 
where a computational graph is extracted from the function to allow transpilation. In most cases this works great, 
but it often doesn't allow dynamic control flow (`if` statements, `while` loops, etc) to be correctly represented in 
the graph - which can cause some kornia functions to not behave as expected when transpiled.


From the Ivy Team
-----------------

We hope you find using Kornia with NumPy, JAX and TensorFlow useful! Ivy is still very much under development, 
so if you find any issues/bugs, feel free to raise an issue on the `ivy <https://github.com/ivy-llc/ivy>`_ repository!

To learn more about Ivy, we recommend you to read through the `Get Started <https://ivy.dev/docs/overview/get_started.html>`_ and 
`Quickstart <https://ivy.dev/docs/demos/quickstart.html>`_ sections of the documentation.

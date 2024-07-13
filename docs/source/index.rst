
.. image:: https://github.com/kornia/data/raw/main/kornia_banner_pixie.png
   :align: center

State-of-the-art and curated Computer Vision algorithms for AI.

Kornia AI is on the mission to leverage and democratize the next generation of Computer Vision tools and Deep Learning libraries
within the context of an Open Source community.

.. code:: python

   >>> import kornia.geometry as K
   >>> registrator = K.ImageRegistrator('similarity')
   >>> model = registrator(img1, img2)

Ready to use with state-of-the art Deep Learning models:

.. code:: python

   >>> import torch.nn as nn
   >>> import kornia.contrib as K
   >>> classifier = nn.Sequential(
   ...   K.VisionTransformer(image_size=224, patch_size=16),
   ...   K.ClassificationHead(num_classes=1000),
   ... )
   >>> logits = classifier(img)    # BxN
   >>> scores = logits.argmax(-1)  # B

Multi-Framework support
-----------------------

Kornia can now be used with frameworks other than PyTorch such as `TensorFlow <https://www.tensorflow.org/>`_, 
`JAX <https://jax.readthedocs.io/en/latest/index.html>`_ and `Numpy <https://numpy.org/>`_ using `Ivy <https://github.com/ivy-llc/ivy>`_. 

In order to use :code:`ivy` to transpile :code:`kornia`, there are a few functions added to the :code:`kornia.transpile` API
1. :code:`kornia.to_tensorflow`
2. :code:`kornia.to_jax`
3. :code:`kornia.to_numpy`
4. :code:`kornia.to_ivy`

Here's an example of using :code:`kornia` with :code:`tf`

.. code:: python

   >>> import tensorflow as tf
   >>> import kornia
   >>> tf_kornia = kornia.to_tensorflow()
   >>> rgb_image = tf.random.normal((1, 3, 224, 224))
   >>> gray_image = kornia_tf.color.rgb_to_grayscale(rgb_image)

:code:`kornia.to_jax`, :code:`kornia.to_numpy` and :code:`kornia.to_ivy` can be used in a 
similar manner. In order to learn more about Ivy, we recommend you to go through `Get Started <https://ivy.dev/docs/overview/get_started.html>`_ and `Quickstart <https://ivy.dev/docs/demos/quickstart.html>`_.
If there are any issues/bugs, feel free to report them on the `ivy <https://github.com/ivy-llc/ivy>`_ repository!

Join the community
------------------

- Join our social network communities with 1.8k+ members:
   - `Twitter <https://twitter.com/kornia_foss>`_: we share the recent research and news for out mainstream community.
   - `Slack <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA>`_: come to us and chat with our engineers and mentors to get support and resolve your questions.
- Subscribe to our `YouTube channel <https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag>`_ to get the latest video demos.

----

.. toctree::
   :caption: GET STARTED
   :hidden:

   get-started/introduction
   get-started/highlights
   get-started/installation
   get-started/about
   Tutorials <https://kornia.github.io/tutorials/>
   get-started/training
   OpenCV AI Kit <https://docs.luxonis.com/en/latest/pages/tutorials/creating-custom-nn-models/#kornia>
   get-started/governance

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2
   :hidden:

   augmentation
   color
   contrib
   core
   enhance
   feature
   filters
   geometry
   sensors
   io
   image
   losses
   metrics
   morphology
   nerf
   tracking
   testing
   utils
   x

.. toctree::
   :caption: KORNIA APPLICATIONS
   :hidden:

   applications/intro
   applications/visual_prompting
   applications/face_detection
   applications/image_augmentations
   applications/image_classification
   applications/image_matching
   applications/image_stitching
   applications/image_registration
   applications/image_denoising
   applications/semantic_segmentation
   applications/object_detection

.. toctree::
   :caption: KORNIA MODELS
   :hidden:

   models/efficient_vit
   models/rt_detr
   models/segment_anything
   models/mobile_sam
   models/yunet
   models/vit
   models/vit_mobile
   models/tiny_vit
   models/loftr
   models/defmo
   models/hardnet
   models/affnet
   models/sold2
   models/dexined

.. toctree::
   :caption: SUPPORT
   :hidden:

   Issue tracker <https://github.com/kornia/kornia/issues>
   Slack community <https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-2AQRi~X9Uu6PLMuUZdvfjA>
   LibreCV community <https://librecv.org>
   Twitter @kornia_foss <https://twitter.com/kornia_foss>
   community/chinese
   Kornia Youtube <https://www.youtube.com/channel/UCI1SE1Ij2Fast5BSKxoa7Ag>
   Kornia LinkedIn <https://www.linkedin.com/company/kornia/>
   Kornia AI <https://kornia.org>

.. toctree::
   :caption: COMMUNITY
   :hidden:

   community/contribute
   community/faqs
   community/bibliography

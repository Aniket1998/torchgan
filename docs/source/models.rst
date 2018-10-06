torchgan.models
===============

.. currentmodule:: torchgan.models

This models subpackage is a collection of popular GAN architectures. It has
the support for existing architectures and provides a base class for
extending to any form of new architecture. Currently the following models
are supported:

.. contents::
    :local:

You can construct a new model by simply calling its constructor.

.. code:: python

    >>> import torchgan.models as models
    >>> dcgan_discriminator = DCGANDiscriminator()
    >>> dcgan_generator = DCGANGenerator()

All models follow the same structure. There are additional customization options.
Look into the individual documentation for such capabilities.

GAN
---
.. autoclass:: Generator
    :members:

    .. automethod:: _weight_initializer
.. autoclass:: Discriminator
    :members:

    .. automethod:: _weight_initializer

DCGAN
-----
.. autoclass:: DCGANGenerator
.. autoclass:: DCGANDiscriminator

Conditional GAN
---------------
.. autoclass:: ConditionalGANGenerator
.. autoclass:: ConditionalGANDiscriminator

InfoGAN
-------
.. autoclass:: InfoGANGenerator
.. autoclass:: InfoGANDiscriminator

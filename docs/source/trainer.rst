torchgan.trainer
================

.. currentmodule:: torchgan.trainer

This subpackage provides ability to perform end to end training capabilities of
the Generator and Discriminator models. It provides strong visualization
capabilities using `tensorboardX <https://github.com/lanpa/tensorboardX>`_. In most cases you will need
to overwrite the :func:`generator_train_iter` and :func:`discriminator_train_iter`.

Currently supported Trainers include:

.. contents::
    :local:

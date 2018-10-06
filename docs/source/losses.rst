torchgan.losses
===============

.. currentmodule:: torchgan.losses

This losses subpackage is a collection of popular loss functions used
in the training of GANS. Currently the following losses are supported:

.. contents::
    :local:

These losses are tested with the current available trainers. So if you need
to implement you custom loss for using with the :mod:`trainer` it is recommended
that you subclass the :class:`GeneratorLoss` and :class:`DiscriminatorLoss`

Loss
----
.. autoclass:: GeneratorLoss
.. autoclass:: DiscriminatorLoss

Least Squares Loss
------------------
.. autoclass:: LeastSquaresGeneratorLoss
    :members:
.. autoclass:: LeastSquaresDiscriminatorLoss
    :members:

Minimax Loss
------------
.. autoclass:: MinimaxGeneratorLoss
    :members:
.. autoclass:: MinimaxDiscriminatorLoss
    :members:

Boundary Equilibrium Loss
-------------------------
.. autoclass:: BoundaryEquilibriumGeneratorLoss
    :members:
.. autoclass:: BoundaryEquilibriumDiscriminatorLoss
    :members:

Energy Based Loss
-----------------
.. autoclass:: EnergyBasedGeneratorLoss
    :members:
.. autoclass:: EnergyBasedDiscriminatorLoss
    :members:

Wasserstein Loss
----------------
.. autoclass:: WassersteinGeneratorLoss
    :members:
.. autoclass:: WassersteinDiscriminatorLoss
    :members:

Mutual Information Penalty
--------------------------
.. autoclass:: MutualInformationPenalty
    :members:

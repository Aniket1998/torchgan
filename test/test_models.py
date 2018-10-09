import unittest
import torch
import torch.distributions as distributions
from sys import path
path.append('..')
from torchgan.models import *


class TestModels(unittest.TestCase):
    def test_dcgan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            gen = DCGANGenerator(encodings[i], channels[i], ch, batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = gen(x)
            assert y.shape == (10, channels[i], 64, 64)

    def test_small_dcgan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            gen = SmallDCGANGenerator(encodings[i], channels[i], ch, batchnorm[i],
                                      nonlinearities[i], last_nonlinearity[i])
            y = gen(x)
            assert y.shape == (10, channels[i], 32, 32)

    def test_dcgan_discriminator(self):
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            x = torch.randn(10, channels[i], 64, 64)
            D = DCGANDiscriminator(channels[i], step[i], batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = D(x)
            assert y.shape == (10, 1, 1, 1)

    def test_small_dcgan_discriminator(self):
        channels = [3, 4]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            x = torch.randn(10, channels[i], 32, 32)
            D = SmallDCGANDiscriminator(channels[i], step[i], batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = D(x)
            assert y.shape == (10, 1, 1, 1)

    def test_conditional_gan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            gen = ConditionalGANGenerator(classes[i], encodings[i], channels[i], ch,
                                          batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = gen(x, torch.rand(10, classes[i]))
            assert y.shape == (10, channels[i], 64, 64)

    def test_conditional_gan_discriminator(self):
        channels = [3, 4]
        classes = [5, 10]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, channels[i], 64, 64)
            gen = ConditionalGANDiscriminator(classes[i], channels[i], ch,
                                              batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = gen(x, torch.rand(10, classes[i]))
            assert y.shape == (10, 1, 1, 1)

    def test_infogan_generator(self):
        encodings = [50, 100]
        channels = [3, 4]
        step = [64, 128]
        dim_cont = [10, 20]
        dim_dis = [30, 40]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            ch = step[i]
            x = torch.randn(10, encodings[i])
            cont = torch.rand(10, dim_cont[i])
            dis = torch.zeros(10, dim_dis[i])
            gen = InfoGANGenerator(dim_cont[i], dim_dis[i], encodings[i], channels[i],
                                   ch, batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y = gen(x, cont, dis)
            assert y.shape == (10, channels[i], 64, 64)

    def test_infogan_discriminator(self):
        channels = [3, 4]
        dim_cont = [10, 20]
        dim_dis = [30, 40]
        step = [64, 128]
        batchnorm = [True, False]
        nonlinearities = [None, torch.nn.ELU(0.5)]
        last_nonlinearity = [None, torch.nn.RReLU(0.25)]
        for i in range(2):
            x = torch.randn(10, channels[i], 64, 64)
            D = InfoGANDiscriminator(dim_dis[i], dim_cont[i], channels[i], step[i],
                                     batchnorm[i], nonlinearities[i], last_nonlinearity[i])
            y, dist_dis, dist_cont = D(x, True)
            assert y.shape == (10, 1, 1, 1)
            assert isinstance(dist_dis, distributions.OneHotCategorical)
            assert isinstance(dist_cont, distributions.Normal)
            assert dist_dis.sample().shape == (10, dim_dis[i])
            assert dist_cont.sample().shape == (10, dim_cont[i])

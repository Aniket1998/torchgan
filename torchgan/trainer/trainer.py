import torch
import torchvision
from warnings import warn
from inspect import signature
from operator import itemgetter
from tensorboardX import SummaryWriter
from ..losses.loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['Trainer']

class Trainer(object):
    # TODO(avik-pal): Needs support of Metrics
    def __init__(self, generator, discriminator, optimizer_generator, optimizer_discriminator,
                 losses_list, device=torch.device("cuda:0"), ndiscriminator=-1, batch_size=128,
                 sample_size=8, epochs=5, checkpoints="./model/gan", retain_checkpoints=5,
                 recon="./images", test_noise=None, log_tensorboard=True, **kwargs):
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        if "optimizer_generator_options" in kwargs:
            self.optimizer_generator = optimizer_generator(self.generator.parameters(),
                                                           **kwargs["optimizer_generator_options"])
        else:
            self.optimizer_generator = optimizer_generator(self.generator.parameters())
        if "optimizer_discriminator_options" in kwargs:
            self.optimizer_discriminator = optimizer_discriminator(self.discriminator.parameters(),
                                                                   **kwargs["optimizer_discriminator_options"])
        else:
            self.optimizer_discriminator = optimizer_discriminator(self.discriminator.parameters())
        self.losses_list = losses_list
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        self.test_noise = torch.randn(self.sample_size, self.generator.encoding_dims, 1, 1,
                                      device=self.device) if test_noise is None else test_noise
        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = torch.randn(1)
        self.real_inputs = torch.randn(1)
        self.labels = torch.randn(1)

        self.loss_information = {
            'generator_losses': [],
            'discriminator_losses': [],
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        self.ndiscriminator = ndiscriminator
        if "loss_information" in kwargs:
            self.loss_information.update(kwargs["loss_information"])
        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        self.writer = SummaryWriter()
        self.log_tensorboard = log_tensorboard
        if self.log_tensorboard:
            self.tensorboard_information = {
                "step": 0,
                "repeat_step": 4,
                "repeats": 1
            }
        self.nrow = kwargs["display_rows"] if "display_rows" in kwargs else 8
        self.labels_provided = kwargs["labels_provided"] if "labels_provided" in kwargs\
                                        else False

    def save_model_extras(self, save_path):
        return {}

    def save_model(self, epoch):
        if self.last_retained_checkpoint == self.retain_checkpoints:
            self.last_retained_checkpoint = 0
        save_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        self.last_retained_checkpoint += 1
        print("Saving Model at '{}'".format(save_path))
        model = {
            'epoch': epoch + 1,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
            'generator_losses': self.loss_information["generator_losses"],
            'discriminator_losses': self.loss_information["discriminator_losses"]
        }
        # FIXME(avik-pal): Not a very good function name
        model.update(self.save_model_extras(save_path))
        torch.save(model, save_path)

    def load_model_extras(self, load_path):
        pass

    def load_model(self, load_path=""):
        if load_path == "":
            load_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        print("Loading Model From '{}'".format(load_path))
        try:
            check = torch.load(load_path)
            self.start_epoch = check['epoch']
            self.generator_losses = check['generator_losses']
            self.discriminator_losses = check['discriminator_losses']
            self.generator.load_state_dict(check['generator'])
            self.discriminator.load_state_dict(check['discriminator'])
            self.optimizer_generator.load_state_dict(check['optimizer_generator'])
            self.optimizer_discriminator.load_state_dict(check['optimizer_discriminator'])
            # FIXME(avik-pal): Not a very good function name
            self.load_model_extras(check)
        except:
            warn("Model could not be loaded from {}. Training from Scratch".format(load_path))
            self.start_epoch = 0
            self.generator_losses = []
            self.discriminator_losses = []

    def _get_step(self, img=False):
        # FIXME(avik-pal): Not a proper step size. Immediately fix this
        if img or self.tensorboard_information["repeats"] < self.tensorboard_information["repeat_step"]:
            self.tensorboard_information["repeats"] += 1
            return self.tensorboard_information["step"]
        else:
            self.tensorboard_information["step"] += 1
            self.tensorboard_information["repeats"] = 1
            return self.tensorboard_information["step"]

    def sample_images(self, epoch):
        save_path = "{}/epoch{}.png".format(self.recon, epoch + 1)
        print("Generating and Saving Images to {}".format(save_path))
        self.generator.eval()
        with torch.no_grad():
            images = self.generator(self.test_noise.to(self.device))
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, save_path, nrow=self.nrow)
            if self.log_tensorboard:
                self.writer.add_image("Generated Samples", img, self._get_step())
        self.generator.train()

    def _verbose_matching(self, verbose):
        # TODO(avik-pal) : better logging strategy
        assert verbose >= 0 and verbose <= 5
        self.niter_print_losses = 10**((6 - verbose) / 2)
        self.save_epoch = 6 - verbose
        self.generate_images = 6 - verbose

    def train_logger(self, running_generator_loss, running_discriminator_loss, epoch, itr=None):
        if itr is None:
            if (epoch + 1) % self.save_epoch == 0 or epoch == self.epochs:
                self.save_model(epoch)
            if (epoch + 1) % self.generate_images or epoch == self.epochs:
                self.sample_images(epoch)
            print("Epoch {} Complete | Mean Generator Loss : {} | Mean Discriminator Loss : {}".format(epoch + 1,
                  running_generator_loss, running_discriminator_loss))
        else:
            print("Epoch {} | Iteration {} | Mean Generator Loss : {} | Mean Discriminator Loss : {}".format(
                  epoch + 1, itr + 1, running_generator_loss, running_discriminator_loss))

    def tensorboard_log(self, running_generator_loss, running_discriminator_loss):
        if self.log_tensorboard:
            self.writer.add_scalar("Discriminator Loss", running_discriminator_loss,
                                                    self._get_step())
            self.writer.add_scalar("Generator Loss", running_generator_loss,
                                                    self._get_step())
            self.writer.add_scalars("Losses", {'Generator Loss': running_generator_loss,
                                               'Discriminator Loss': running_discriminator_loss},
                                    self._get_step())

    def _get_argument_maps(self, loss):
        sig = signature(loss.train_ops)
        args = list(sig.parameters.keys())
        for arg in args:
            if arg not in self.__dict__:
                raise Exception("Argument : %s needed for Loss not present".format(arg))
        return args

    def _store_loss_maps(self):
        self.loss_arg_maps = []
        for loss in self.losses_list:
            self.loss_arg_maps.append(self._get_argument_maps(loss))

    def train_stopper(self):
        if self.ndiscriminator == -1:
            return False
        else:
            return self.loss_information["discriminator_iters"] % self.ndiscriminator != 0

    def train_iter_custom(self):
        pass

    def train_iter(self):
        self.train_iter_custom()
        ldis = 0.0
        lgen = 0.0
        gen_iter = 0
        dis_iter = 0
        for i, loss in enumerate(self.losses_list):
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[i])(self.__dict__))
                if type(cur_loss) is tuple:
                    ldis += cur_loss[1]
                    lgen += cur_loss[0]
                    gen_iter, dis_iter = 1, 1
                else:
                    ldis += cur_loss
                    dis_iter = 1
            elif isinstance(loss, GeneratorLoss):
                if self.ndiscriminator == -1 or\
                   self.loss_information["discriminator_iters"] % self.ncritic == 0:
                    lgen += loss.train_ops(*itemgetter(*self.loss_arg_maps[i])(self.__dict__))
                    gen_iter = 1
            elif isinstance(loss, DiscriminatorLoss):
                ldis += loss.train_ops(*itemgetter(*self.loss_arg_maps[i])(self.__dict__))
                dis_iter = 1
        return ldis, lgen, dis_iter, gen_iter

    def train(self, data_loader, **kwargs):
        self.generator.train()
        self.discriminator.train()

        running_generator_loss = 0.0
        running_discriminator_loss = 0.0

        for epoch in range(self.start_epoch, self.epochs):

            for images, labels in data_loader:

                if not images.size()[0] == self.batch_size:
                    continue

                self.real_inputs = images.to(self.device)
                self.labels = labels.to(self.device)
                self.noise = torch.randn(self.batch_size, self.generator.encoding_dims, 1, 1,
                                         device=self.device)

                ldis, lgen, dis_iter, gen_iter = self.train_iter()
                self.loss_information['generator_losses'].append(lgen)
                self.loss_information['discriminator_losses'].append(ldis)
                self.loss_information['generator_iters'] += gen_iter
                self.loss_information['discriminator_iters'] += dis_iter
                running_discriminator_loss += self.loss_information['discriminator_losses'][-1]
                running_generator_loss += self.loss_information['generator_losses'][-1]

                self.tensorboard_log(running_generator_loss / self.loss_information['generator_iters'],
                    running_discriminator_loss / self.loss_information['discriminator_iters'])

                if self.train_stopper():
                    break

                if self.loss_information['discriminator_iters'] % self.niter_print_losses == 0 \
                   and not self.loss_information['discriminator_iters'] == 0:
                    self.train_logger(running_generator_loss / self.loss_information['generator_iters'],
                                      running_discriminator_loss / self.loss_information['discriminator_iters'],
                                      epoch, self.loss_information['discriminator_iters'])

            self.train_logger(running_generator_loss / self.loss_information['generator_iters'],
                              running_discriminator_loss / self.loss_information['discriminator_iters'],
                              epoch)

        print("Training of the Model is Complete")

    def __call__(self, data_loader, verbose=1, **kwargs):
        self._verbose_matching(verbose)
        self._store_loss_maps()
        self.train(data_loader, **kwargs)
        self.writer.close()

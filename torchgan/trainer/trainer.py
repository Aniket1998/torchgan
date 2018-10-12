import torch
import torchvision
from warnings import warn
from inspect import signature
from operator import itemgetter
from tensorboardX import SummaryWriter
from ..losses.loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['Trainer']

class Trainer(object):
    def __init__(self, generator, discriminator, optimizer_generator, optimizer_discriminator,
                 losses_list, metrics_list=None, device=torch.device("cuda:0"), ndiscriminator=-1, batch_size=128,
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
        self.losses = {}
        self.loss_logs = {}
        for loss in losses_list:
            name = type(loss).__name__
            self.loss_logs[name] = []
            self.losses[name] = loss
        if metrics_list is None:
            self.metrics = None
            self.metric_logs = None
        else:
            self.metric_logs = {}
            self.metrics = {}
            for metric in metrics_list:
                name = type(metric).__name__
                self.metric_logs[name] = []
                self.metrics[name] = metric
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        self.test_noise = torch.randn(self.sample_size, self.generator.encoding_dims,
                                      device=self.device) if test_noise is None else test_noise
        # Not needed but we need to store this to avoid errors. Also makes life simpler
        self.noise = torch.randn(1)
        self.real_inputs = torch.randn(1)
        self.labels = torch.randn(1)

        self.loss_information = {
            'generator_losses': 0.0,
            'discriminator_losses': 0.0,
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        self.ndiscriminator = ndiscriminator
        if "loss_information" in kwargs:
            self.loss_information.update(kwargs["loss_information"])
        if "loss_logs" in kwargs:
            self.loss_logs.update(kwargs["loss_logs"])
        if "metric_logs" in kwargs:
            self.metric_logs.update(kwargs["metric_logs"])
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
            'loss_information': self.loss_information,
            'loss_objects': self.losses,
            'metric_objects': self.metrics,
            'loss_logs': self.loss_logs,
            'metric_logs': self.metric_logs
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
            self.losses = check['loss_objects']
            self.metrics = check['metric_objects']
            self.loss_information = check['loss_information']
            self.loss_logs = check['loss_logs']
            self.metric_logs = check['metric_logs']
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

    # TODO(avik-pal): The _get_step will fail in a lot of cases
    def _get_step(self, update=True):
        if not update:
            return self.tensorboard_information["step"]
        if self.tensorboard_information["repeats"] < self.tensorboard_information["repeat_step"]:
            self.tensorboard_information["repeats"] += 1
            return self.tensorboard_information["step"]
        else:
            self.tensorboard_information["step"] += 1
            self.tensorboard_information["repeats"] = 1
            return self.tensorboard_information["step"]

    def sample_images(self, epoch):
        save_path = "{}/epoch{}.png".format(self.recon, epoch + 1)
        print("Generating and Saving Images to {}".format(save_path))
        with torch.no_grad():
            images = self.generator(self.test_noise.to(self.device))
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, save_path, nrow=self.nrow)
            if self.log_tensorboard:
                self.writer.add_image("Generated Samples", img, self._get_step(False))

    def train_logger(self, epoch, running_losses):
        print('Epoch {} Summary: '.format(epoch + 1))
        for name, val in running_losses.items():
            print('Mean {} : {}'.format(name, val))

    def tensorboard_log_losses(self):
        if self.log_tensorboard:
            running_generator_loss = self.loss_information["generator_losses"] /\
                self.loss_information["generator_iters"]
            running_discriminator_loss = self.loss_information["discriminator_losses"] /\
                self.loss_information["discriminator_iters"]
            self.writer.add_scalar("Running Discriminator Loss",
                                   running_discriminator_loss,
                                   self._get_step())
            self.writer.add_scalar("Running Generator Loss",
                                   running_generator_loss,
                                   self._get_step())
            self.writer.add_scalars("Running Losses",
                                   {"Running Discriminator Loss": running_discriminator_loss,
                                    "Running Generator Loss": running_generator_loss},
                                   self._get_step())

    def tensorboard_log_metrics(self):
        if self.tensorboard_log:
            for name, value in self.loss_logs.items():
                if type(value) is tuple:
                    self.writer.add_scalar('Losses/{}-Generator'.format(name), value[0], self._get_step(False))
                    self.writer.add_scalar('Losses/{}-Discriminator'.format(name), value[1], self._get_step(False))
                else:
                    self.writer.add_scalar('Losses/{}'.format(name), value, self._get_step(False))
            if self.metric_logs:
                for name, value in self.metric_logs.items():
                    # FIXME(Aniket1998): Metrics step should be number of epochs so far
                    self.writer.add_scalar("Metrics/{}".format(name),
                                           value, self._get_step(False))

    def _get_argument_maps(self, loss):
        sig = signature(loss.train_ops)
        args = list(sig.parameters.keys())
        for arg in args:
            if arg not in self.__dict__:
                raise Exception("Argument : %s needed for %s not present".format(arg, type(loss).__name__))
        return args

    def _store_loss_maps(self):
        self.loss_arg_maps = {}
        for name, loss in self.losses.items():
            self.loss_arg_maps[name] = self._get_argument_maps(loss)

    def train_stopper(self):
        if self.ndiscriminator == -1:
            return False
        else:
            return self.loss_information["discriminator_iters"] % self.ndiscriminator != 0

    def train_iter_custom(self):
        pass

    # TODO(avik-pal): Clean up this function and avoid returning values
    def train_iter(self):
        self.train_iter_custom()
        ldis, lgen, dis_iter, gen_iter = 0.0, 0.0, 0, 0
        for name, loss in self.losses.items():
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                self.loss_logs[name].append(cur_loss)
                if type(cur_loss) is tuple:
                    lgen, ldis, gen_iter, dis_iter = lgen + cur_loss[0], ldis + cur_loss[1],\
                        gen_iter + 1, dis_iter + 1
            elif isinstance(loss, GeneratorLoss):
                if self.ndiscriminator == -1 or\
                   self.loss_information["discriminator_iters"] % self.ncritic == 0:
                    cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                    self.loss_logs[name].append(cur_loss)
                    lgen, gen_iter = lgen + cur_loss, gen_iter + 1
            elif isinstance(loss, DiscriminatorLoss):
                cur_loss = loss.train_ops(*itemgetter(*self.loss_arg_maps[name])(self.__dict__))
                self.loss_logs[name].append(cur_loss)
                ldis, dis_iter = ldis + cur_loss, dis_iter + 1
        return lgen, ldis, gen_iter, dis_iter

    def log_metrics(self, epoch):
        if not self.metric_logs:
            warn('No evaluation metric logs present')
        else:
            for name, val in self.metric_logs.item():
                print('{} : {}'.format(name, val))
            self.tensorboard_log_metrics()

    def eval_ops(self, epoch, **kwargs):
        self.sample_images(epoch)
        if self.metrics is not None:
            for name, metric in self.metrics.items():
                if name + '_inputs' not in kwargs:
                    raise Exception("Inputs not provided for metric {}".format(name))
                else:
                    self.metric_logs[name].append(metric.metric_ops(self.generator,
                                                                    self.discriminator, kwargs[name + '_inputs']))
                    self.log_metrics(self, epoch)

    def train(self, data_loader, **kwargs):
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.start_epoch, self.epochs):
            self.generator.train()
            self.discriminator.train()
            for data in data_loader:
                if type(data) is tuple:
                    if not data[0].size()[0] == self.batch_size:
                        continue
                    self.real_inputs = data[0].to(self.device)
                    self.labels = data[1].to(self.device)
                else:
                    if not data.size()[0] == self.batch_size:
                        continue
                    self.real_inputs = data[0].to(self.device)

                self.noise = torch.randn(self.batch_size, self.generator.encoding_dims,
                                         device=self.device)

                lgen, ldis, gen_iter, dis_iter = self.train_iter()
                self.loss_information['generator_losses'] += lgen
                self.loss_information['discriminator_losses'] += ldis
                self.loss_information['generator_iters'] += gen_iter
                self.loss_information['discriminator_iters'] += dis_iter

                self.tensorboard_log_losses()

                if self.train_stopper():
                    break

            self.save_model(epoch)
            self.train_logger(epoch,
                              {'Generator Loss': self.loss_information['generator_losses'] /
                              self.loss_information['generator_iters'],
                              'Discriminator Loss': self.loss_information['discriminator_losses'] /
                              self.loss_information['discriminator_iters']})
            self.generator.eval()
            self.discriminator.eval()
            self.eval_ops(epoch, **kwargs)

        print("Training of the Model is Complete")

    def __call__(self, data_loader, **kwargs):
        self._store_loss_maps()
        self.train(data_loader, **kwargs)
        self.writer.close()

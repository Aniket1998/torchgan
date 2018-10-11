import torch
import torchvision
from warnings import warn
from ..losses import GeneratorLoss, DiscriminatorLoss

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, generator, discriminator, optimGen, optimDis, losses, metrics=None,
                 device=torch.device('cuda:0'), batch_size=128, sample_size=8, epochs=20,
                 checkpoints='./model/gan.pth', retain_checkpoints=5, sample_path='./images',
                 test_noise=None, log_tensorboard=True, **kwargs):
        self.device = device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        if 'optim_generator_options' in kwargs:
            self.optimGen = optimGen(generator.parameters(), kwargs['optim_generator_options'])
        else:
            self.optimGen = optimGen(generator.parameters())

        if 'optim_discriminator_options' in kwargs:
            self.optimDis = optimDis(discriminator.parameters(), kwargs['optim_discriminator_options'])
        else:
            self.optimDis = optimDis(discriminator.parameters())

        self.combined_loss_obj = {}
        self.generator_loss_obj = {}
        self.discriminator_loss_obj = {}
        self.loss_logs = {}
        self.metric_logs = {}
        for loss in losses:
            self.loss_logs[type(loss).__name__] = []
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                self.combined_loss_obj[type(loss).__name__] = loss
            elif isinstance(loss, GeneratorLoss):
                self.generator_loss_obj[type(loss).__name__] = loss
            else:
                self.discriminator_loss_obj[type(loss).__name__] = loss

        if metrics is None:
            self.metric_obj = None
            self.metric_logs = None
        else:
            self.metric_obj = {}
            self.metric_logs = {}
            for metric in metrics:
                self.metric_logs[type(metric).__name__] = []
                self.metric_obj[type(metric).__name__] = metric

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.sample_path = sample_path
        self.log_tensorboard = log_tensorboard
        self.test_noise = torch.randn(self.sample_size, self.encoding_dims) if test_noise is None else test_noise

        self.start_epoch = 0
        self.last_retained_checkpoint = 0
        if 'display_rows' not in kwargs:
            self.nrow = 8
        else:
            self.nrow = kwargs['display_rows']

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
            'combined_loss_obj': self.combined_loss_obj,
            'discriminator_loss_obj': self.discriminator_loss_obj,
            'generator_loss_obj': self.generator_loss_obj,
            'loss_logs': self.loss_logs,
            'metric_obj': self.metric_obj,
            'metric_logs': self.metric_logs
        }
        torch.save(model, save_path)

    def load_model(self, load_path=''):
        if load_path == '':
            load_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        print("Loading Model From '{}'".format(load_path))
        try:
            check = torch.load(load_path)
            self.start_epoch = check['epoch']
            self.combined_loss_obj = check['combined_loss_obj']
            self.generator_loss_obj = check['generator_loss_obj']
            self.discriminator_loss_obj = check['discriminator_loss_obj']
            self.loss_logs = check['loss_logs']
            self.metric_obj = check['metric_obj']
            self.metric_logs = check['metric_logs']
            self.generator.load_state_dict(check['generator'])
            self.discriminator.load_state_dict(check['discriminator'])
            self.optimizer_generator.load_state_dict(check['optimizer_generator'])
            self.optimizer_discriminator.load_state_dict(check['optimizer_discriminator'])
        except:
            warn("Model could not be loaded from {}. Training from Scratch".format(load_path))
            self.start_epoch = 0
            self.generator_losses = []
            self.discriminator_losses = []

    def sample_images(self, epoch):
        save_path = "{}/epoch{}.png".format(self.recon, epoch + 1)
        print("Generating and Saving Images to {}".format(save_path))
        self.generator.eval()
        with torch.no_grad():
            images = self.generator(self.test_noise.to(self.device))
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, save_path, nrow=self.nrow)
        self.generator.train()

    def train_logger(self, epoch, running_losses):
        print('Epoch {} Summary: '.format(epoch + 1))
        for name, val in running_losses.items():
            print('Mean {} : {}'.format(name, val))

    def eval_ops(self, epoch, running_losses):
        self.sample_images(epoch)
        for name, val in running_losses.items():
            self.loss_logs[name].append(val)

    def training_loop(self, device_input, running_losses, has_labels):
        for name, obj in self.generator_loss_obj.keys():
            running_losses[name] += obj.train_ops(self.generator, self.discriminator,
                                                  self.optimDis, device_input, has_labels)
        for name, obj in self.discriminator_loss_obj.keys():
                running_losses[name] += obj.train_ops(self.generator, self.discriminator,
                                                      self.optimDis, device_input, has_labels)
        for name, obj in self.combined_loss_obj.keys():
            running_losses[name] += obj.train_ops(self.generator, self.discriminator, self.optimGen, self.optimDis,
                                                  device_input, has_labels)

    # TODO(Aniket1998): Add support for multiple iterations of the discriminator
    def train(self, data_loader, has_labels=False, **kwargs):
        for epoch in range(self.start_epoch, self.epochs):
            self.generator.train()
            self.discriminator.train()
            # TODO(Aniket1998): Find faster alternatives to creating this dictionary every epoch
            running_losses = {}
            for name in self.loss_logs.keys():
                running_losses[name] = 0.0

            for i, inputs in enumerate(data_loader, 1):
                device_input = []
                for i in inputs:
                    device_input.append(i.to(self.device))
                self.training_loop(device_input, running_losses, has_labels)

            for key in running_losses.keys():
                running_losses[key] /= i

            self.generator.eval()
            self.discriminator.eval()
            self.train_logger(epoch, running_losses)
            self.eval_ops(epoch, running_losses)

        print('Training is complete')

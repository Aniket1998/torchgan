import torch
import torchvision
from warnings import warn

__all__ = ['Trainer']

class Trainer(object):
    def __init__(self, generator, discriminator, optimizer_generator, optimizer_discriminator,
                 generator_loss, discriminator_loss, device=torch.device("cuda:0"),
                 batch_size=128, sample_size=8, epochs=5, checkpoints="./model/gan",
                 retain_checkpoints=5, recon="./images/", test_noise=None, **kwargs):
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
        if "loss_generator_options" in kwargs:
            self.generator_loss = generator_loss(**kwargs["loss_generator_options"])
        else:
            self.generator_loss = generator_loss()
        if "loss_discriminator_options" in kwargs:
            self.discriminator_loss = discriminator_loss(**kwargs["loss_discriminator_options"])
        else:
            self.discriminator_loss = discriminator_loss()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        self.test_noise = torch.randn(self.sample_size, self.generator.encoding_dims, 1, 1,
                                      device=self.device) if test_noise is None else test_noise
        self.loss_information = {
            'generator_losses': [],
            'discriminator_loss': [],
            'generator_iters': 0,
            'discriminator_iters': 0,
        }
        if "loss_information" in kwargs:
            self.loss_information.update(kwargs["loss_information"])
        if not "target_dim" in kwargs:
            target_dim = 1
        else:
            target_dim = kwargs["target_dim"]
        self.targets = {
            'discriminator_target_real': torch.ones(self.batch_size, target_dim, device=self.device).squeeze(),
            'discriminator_target_fake': torch.zeros(self.batch_size, target_dim, device=self.device).squeeze()
        }
        self.start_epoch = 0
        self.last_retained_checkpoint = 0

    def save_model_extras(self, save_path):
        return {}

    def save_model(self, epoch):
        if self.last_retained_checkpoint == self.retain_checkpoints:
            self.last_retained_checkpoint = 0
        save_path = self.checkpoints + str(self.last_retained_checkpoint) + '.model'
        print("Saving Model at '{}'".format(save_path))
        model = {
            'epoch': epoch + 1,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_generator': self.optimizer_generator.state_dict(),
            'optimizer_discriminator': self.optimizer_discriminator.state_dict(),
            'generator_losses': self.generator_losses,
            'discriminator_losses': self.discriminator_losses
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

    def sample_images(self, epoch, nrow=8):
        save_path = "{}/epoch{}.png".format(self.recon, epoch + 1)
        print("Generating and Saving Images to {}".format(save_path))
        self.generator.eval()
        with torch.no_grad():
            images = self.generator(self.test_noise.to(self.device))
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, save_path, nrow=nrow)
        self.generator.train()

    def _verbose_matching(self, verbose):
        assert verbose >= 0 and verbose <= 5
        self.save_iter = 10**((6 - verbose) / 2)
        self.save_epoch = 6 - verbose
        self.generate_images = 6 - verbose

    def train_logger(self, running_generator_loss, running_discriminator_loss, epoch, itr=None):
        if itr is None:
            if (epoch + 1) % self.save_epoch == 0 or epoch == self.epochs:
                self.save_model(epoch)
            if (epoch + 1) % self.generate_images or epoch == self.epochs:
                self.sample_images(epoch)
            print("Epoch {} Complete | Mean Generator Loss : {} | Mean Discriminator Loss : {}".format(epoch + 1,
                  running_generator_loss, running_generator_loss))
        else:
            print("Epoch {} | Iteration {} | Mean Generator Loss : {} | Mean Discriminator Loss : {}".format(
                  epoch + 1, itr + 1, running_generator_loss, running_discriminator_loss))

    def train_stopper(self):
        return False

    def generator_train_iter(self, **kwargs):
        sampled_noise = torch.randn(self.batch_size, self.generator.encoding_dims, 1, 1, device=self.device)
        g_loss = self.generator_loss(self.discriminator(self.generator(sampled_noise)))
        g_loss.backward()
        self.loss_information['generator_losses'].append(g_loss)
        self.loss_information['generator_iters'] += 1

    def discriminator_train_iter(self, images, labels, **kwargs):
        sampled_noise = torch.randn(self.batch_size, self.generator.encoding_dims, 1, 1, device=self.device)
        d_real = self.discriminator(images).squeeze()
        d_loss_real = self.discriminator_loss(d_real, self.targets["discriminator_target_real"])
        d_fake = self.discriminator(self.generator(sampled_noise).detach()).squeeze()
        d_loss_fake = self.discriminator_loss(d_fake, self.targets["discriminator_target_fake"])
        d_loss = d_loss_fake + d_loss_real
        d_loss.backward()
        self.loss_information['discriminator_losses'].append(d_loss)
        self.loss_information['discriminator_iters'] += 1

    def train(self, data_loader, **kwargs):
        self.generator.train()
        self.discriminator.train()

        generator_options = {}
        discriminator_options = {}

        if "discriminator_options" in kwargs:
            discriminator_options = kwargs["discriminator_options"]
        if "generator_options" in kwargs:
            generator_options = kwargs["generator_options"]

        for epoch in range(self.start_epoch, self.epochs):

            running_generator_loss = 0.0
            running_discriminator_loss = 0.0

            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                self.discriminator_train_iter(images, labels, **discriminator_options)
                self.optimizer_discriminator.step()
                running_discriminator_loss += self.loss_information['discriminator_losses'][-1]

                self.generator.zero_grad()
                self.generator_train_iter(**generator_options)
                self.optimizer_generator.step()
                running_generator_loss += self.loss_information['generator_losses'][-1]

                # NOTE(avik-pal): A small hack to support WGAN
                if self.train_stopper():
                    break

                if self.loss_information['discriminator_iters'] % self.niter_print_losses == 0:
                    # FIXME(avik-pal): Sadly the iteration printed will be the discriminator iters
                    self.train_logger(running_generator_loss / self.loss_information['generator_iters'],
                                      running_discriminator_loss / self.loss_information['discriminator_iters'],
                                      self.loss_information['discriminator_iters'])

            self.train_logger(running_generator_loss / self.loss_information['generator_iters'],
                              running_discriminator_loss / self.loss_information['discriminator_iters'])

        print("Training of the Model is Complete")

    def __call__(self, data_loader, verbose=1, **kwargs):
        self._verbose_matching(verbose)
        self.train(data_loader, **kwargs)

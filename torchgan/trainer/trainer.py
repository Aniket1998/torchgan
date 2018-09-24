import torch
import torchvision
from torchgan.utils import get_default_lr

class Trainer(object):
    def __init__(self, device, generator, discriminator, optimizer_generator,
                 optimizer_discriminator, lr_generator, lr_discriminator, batch_size,
                 sample_size, epochs, checkpoints, retain_checkpoints, recon):
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_generator
        if lr_generator == -1:
            self.lr_generator = get_default_lr(self.optimizer_generator)
        if lr_discriminator == -1:
            self.lr_discriminator = get_default_lr(self.optimizer_discriminator)
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.epochs = epochs
        self.checkpoints = checkpoints
        self.retain_checkpoints = retain_checkpoints
        self.recon = recon
        self.generator_losses = []
        self.discriminator_losses = []
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
            # TODO(avik-pal): Replace this message by a warning
            print("Model could not be loaded from {}. Training from Scratch".format(load_path))
            self.start_epoch = 0
            self.generator_losses = []
            self.discriminator_losses = []

    def sample_images(self, epoch, nrow=8):
        with torch.no_grad():
            images = self.generator(self.test_noise)
            img = torchvision.utils.make_grid(images)
            torchvision.utils.save_image(img, "{}/epoch{}.png".format(self.recon, epoch+1), nrow=nrow)

    def train_logger(self):
        pass

    def train(self):
        pass

    def train_loop(self):
        pass

    def __call__(self):
        pass

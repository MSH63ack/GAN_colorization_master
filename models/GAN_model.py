import torch
from torch import nn
from .GAN_lose import GANLoss
from utiles.utiles import init_model
from .ResNetDiscriminator import ResNetDiscriminator
from .Unet import Unet
from torch.optim import Adam


class GANModel(nn.Module):
    """
    The main GAN model that combines both a generator and a discriminator,
    backpropagation and forward propagation functions. This class contains all the basic network logic.
    """
    def __init__(self, model_generator=None, lr_generator=2e-4, lr_discriminator=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        """
        :param model_generator: Input model generator object
        :param lr_generator:
        :param lr_discriminator:
        :param beta1:
        :param beta2:
        :param lambda_L1:
        """
        super().__init__()

        self.loss_generator = None
        self.loss_generator_L1 = None
        self.loss_generator_GAN = None
        self.ab = None
        self.L = None
        self.loss_discriminator_fake = None
        self.loss_discriminator = None
        self.loss_discriminator_real = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if model_generator is None:
            self.model_generator = init_model(Unet(in_channels=1, out_channels=2,
                                                   n_down=8, num_filters=64), self.device)
        else:
            self.model_generator = model_generator.to(self.device)

        self.model_discriminator = init_model(ResNetDiscriminator(input_channels=3, n_down=3, num_filters=64),
                                              self.device)

        self.GAN_lose = GANLoss().to(self.device)
        self.L1criterion = nn.L1Loss()
        self.optim_generator = Adam(self.model_generator.parameters(), lr=lr_generator, betas=(beta1, beta2))
        self.optim_discriminator = Adam(self.model_discriminator.parameters(),
                                        lr=lr_discriminator, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """
        Set for model validation and not train parameters grad as True.
        :param model: DL torch model
        :param requires_grad: Grad value. Default: True
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        """
        Split input dict data object into two inputs files for generator and discriminator
        :param data: Input dict model data
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.model_generator(self.L)

    def backward_discriminator(self):
        """
        backward propagation for discriminator
        :return:
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.model_discriminator(fake_image.detach())
        self.loss_discriminator_fake = self.GAN_lose(fake_predictions, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_predictions = self.model_discriminator(real_image)
        self.loss_discriminator_real = self.GAN_lose(real_predictions, True)
        self.loss_discriminator = (self.loss_discriminator_fake + self.loss_discriminator_real) * 0.5
        self.loss_discriminator.backward()

    def backward_generator(self):
        """
        backward propagation for generator
        :return:
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.model_discriminator(fake_image)
        self.loss_generator_GAN = self.GAN_lose(fake_predictions, True)
        self.loss_generator_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_generator = self.loss_generator_GAN + self.loss_generator_L1
        self.loss_generator.backward()

    def optimize(self):
        self.forward()
        self.model_discriminator.train()
        self.set_requires_grad(self.model_discriminator, True)
        self.optim_discriminator.zero_grad()
        self.backward_discriminator()
        self.optim_discriminator.step()

        self.model_generator.train()
        self.set_requires_grad(self.model_generator, False)
        self.optim_generator.zero_grad()
        self.backward_generator()
        self.optim_generator.step()

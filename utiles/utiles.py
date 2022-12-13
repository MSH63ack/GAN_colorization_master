from torch import nn
import torch
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import time


def init_weights(model, init='norm', gain=0.02):
    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(module.weight.data, mean=0.0, std=gain)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(module.weight.data, 1., gain)
            nn.init.constant_(module.bias.data, 0.)

    model.apply(init_func)
    print(f"model initialized with {init} initialization")
    return model


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


class AverageMetrics:
    """
    Class for calculate mean GAN metrics
    """
    def __init__(self):
        self.reset()
        self.sum = 0.
        self.avg = 0
        self.count = 0

    def reset(self):
        self.count = 0.
        self.avg = 0.
        self.sum = 0.

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_metrics():
    """
    Helper function for creat loss metrics result
    :return: dict { loss_discriminator_fake: mean loss discriminator fake value,
                    loss_discriminator_real': mean loss discriminator real,
                    loss_discriminator': mean loss discriminator,
                    loss_generator_GAN': mean loss generator GAN,
                    loss_generator_L1': mean loss generator L1,
                    loss_generator': mean loss generator }
    """
    loss_discriminator_fake = AverageMetrics()
    loss_discriminator_real = AverageMetrics()
    loss_discriminator = AverageMetrics()
    loss_generator_GAN = AverageMetrics()
    loss_generator_L1 = AverageMetrics()
    loss_generator = AverageMetrics()

    return {'loss_discriminator_fake': loss_discriminator_fake,
            'loss_discriminator_real': loss_discriminator_real,
            'loss_discriminator': loss_discriminator,
            'loss_generator_GAN': loss_generator_GAN,
            'loss_generator_L1': loss_generator_L1,
            'loss_generator': loss_generator}


def update_losses(model, loss_metric_dict, count):
    """
    Helper function for updating losses information
    :param model: Input DL model
    :param loss_metric_dict: Loss metric dictionary
    :param count: count
    :return:
    """
    for loss_name, loss_meter in loss_metric_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    """
    Helper function for visualize model prediction
    :param model: Input model GAN
    :param data: Input data images
    :param save: Save images results local. Default: True
    :return:
    """
    model.model_generator.eval()

    with torch.no_grad():
        model.setup_input(data)
        model.forward()

    model.model_generator.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    """
    Helper function for logging model train process
    :param loss_meter_dict:
    :return:
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

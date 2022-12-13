import torch
from torch import nn
from torchsummary import summary
from torch.nn import Conv2d
from torch.nn import Sequential
from torch.nn import BatchNorm2d
from torch.nn import LeakyReLU


class ResNetDiscriminator(nn.Module):
    """
    Neural network based on ResNet architecture
    """
    def __init__(self, input_channels, num_filters=64, n_down=3):
        super().__init__()

        model = [self.get_layers(input_channels, num_filters, is_normalization=False)]
        model += [self.get_layers(
            num_filters * 2 ** i, num_filters * 2 ** (i + 1), stride=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, stride=1, is_normalization=False,
                                  is_activation=False)]

        self.model = Sequential(*model)

    def get_layers(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, is_normalization=True,
                   is_activation=True):
        """
        An auxiliary function for creating layers in a neural network
        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size:  Size of the convolutional kernel. Default: 4
        :param stride: Stride of the convolution. Default: 2
        :param padding: Padding added to all four sides of the input. Default: 1
        :param is_normalization: Add BatchNorm2d layer if True. Default: True
        :param is_activation: Add LeakyReLU activation function if True. Default: True
        :return: Model layers sequential
        """
        layers = [Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not is_normalization)]
        if is_normalization:
            layers += [BatchNorm2d(out_channels)]
        if is_activation:
            layers += [LeakyReLU(0.2, True)]
        return Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input value for model
        :return: Output model prediction
        """
        return self.model(x)


if __name__ == "__main__":
    model = ResNetDiscriminator(input_channels=3)
    print(summary(model, (3, 256, 256)))

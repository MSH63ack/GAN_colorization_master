import torch
from torch import nn
from torchsummary import summary
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
from torch.nn import LeakyReLU


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None:
            input_c = nf
        downconv = Conv2d(input_c, ni, kernel_size=4,
                          stride=2, padding=1, bias=False)
        downrelu = LeakyReLU(0.2, True)
        downnorm = BatchNorm2d(ni)
        uprelu = ReLU(True)
        upnorm = BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(out_channels, out_filters, input_c=in_channels, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Unet(in_channels=1, out_channels=2, n_down=8, num_filters=64)
    print(summary(model, (1, 256, 256)))




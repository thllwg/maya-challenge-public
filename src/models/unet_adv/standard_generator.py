import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import custom_conv_layer


def double_conv(in_channels, out_channels, activation, norm_layer, pool=False):
    layer = []
    if pool:
        layer.append(nn.MaxPool2d(2, ceil_mode=True))
    layer.extend([
        custom_conv_layer(in_channels, out_channels, act_fn=activation, norm_layer=norm_layer),
        custom_conv_layer(out_channels, out_channels, act_fn=activation, norm_layer=norm_layer),
    ])
    return nn.Sequential(*layer)


class StandardNet(torch.nn.Module):
    def __init__(self, c_in: int, act_fn: nn.Module, norm_layer: nn.Module):
        super(StandardNet, self).__init__()
        depths = [c_in, 64, 128, 256, 512, 1024]
        layers = [
            double_conv(depths[0], depths[1], act_fn, norm_layer, pool=False),
            double_conv(depths[1], depths[2], act_fn, norm_layer, pool=True),
            double_conv(depths[2], depths[3], act_fn, norm_layer, pool=True),
            double_conv(depths[3], depths[4], act_fn, norm_layer, pool=True),
            double_conv(depths[4], depths[5], act_fn, norm_layer, pool=True),
        ]

        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(F.dropout2d(x, .1, self.training))
            out.append(x)
        return out

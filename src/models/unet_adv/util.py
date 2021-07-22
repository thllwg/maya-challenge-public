import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm, weight_norm

from src.models.unet_adv.swin_generator import SwinTransformer


def noop(x): return x


def get_act(name):
    if name == "elu":
        return nn.ELU(inplace=True, alpha=0.54)
    elif name == "gelu":
        return nn.GELU()
    elif name in ["leakyrelu", "leaky_relu"]:
        return nn.LeakyReLU(0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_norm_layer(name):
    name = name.lower()
    if name in ["bn", "batch", "batchnorm"]:
        return norm_layer_bn
    elif name in ["gn", "group", "groupnorm"]:
        return norm_layer_gn
    elif name in ["no", "none", "spectral", "weight"]:
        return lambda x: nn.Sequential()
    else:
        raise NotImplementedError(f"{name} is not implemented")


def norm_layer_bn(dims):
    return nn.BatchNorm2d(dims)


def norm_layer_gn(dims):
    d = 32
    while d > 0:
        if dims % d == 0:
            return nn.GroupNorm(d, dims)
        d -= 1


def icnr_init(x, scale=2):
    "ICNR init of `x`, with `scale` and `init` function."
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = nn.init.xavier_normal_(torch.zeros([ni2, nf, h, w]), gain=1.55).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    return k


def init_weights(m):
    if isinstance(m, SwinTransformer):
        return
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight, gain=1.55)
        if m.bias is not None:
            m.bias.data.fill_(1e-3)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

    if isinstance(m, CustomPixelShuffle_ICNR):
        m.conv[0].weight.data.copy_(icnr_init(m.conv[0].weight.data))


def custom_conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias: bool = None,
                      norm_type: str = "batch", norm_layer: nn.Module = nn.BatchNorm2d, act_fn: nn.Module = None,
                      self_attention: bool = False, extra_bn: bool = False, groups: int = 1):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks - 1) // 2
    bn = norm_type in ("batch", "batch_zero", "bn", "gn", "group", "groupnorm") or extra_bn is True
    if bias is None: bias = not bn
    conv = nn.Conv2d(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, groups=groups)

    if norm_type == "weight":
        conv = weight_norm(conv)
    elif norm_type == "spectral":
        conv = spectral_norm(conv)
    layers = [conv]
    if act_fn is not None: layers.append(act_fn)
    if bn: layers.append(norm_layer(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class CustomPixelShuffle_ICNR(nn.Module):
    ''''
    Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`.
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/layers.py#L374
    '''

    def __init__(self, ni: int, act_fn: nn.Module, nf: int = None, scale: int = 2, blur: bool = False, **kwargs):
        super().__init__()
        nf = ni if nf is None else nf
        kwargs["norm_type"] = "weight"
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, **kwargs)
        self.do_blur = blur
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = act_fn

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


class SelfAttention(nn.Module):
    "Self attention layer for nd."

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

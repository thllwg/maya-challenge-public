import torch
from torch import nn as nn
from torch.nn import functional as F

from src.models.unet.util import get_norm_layer, get_act


class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Originally implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            num_classes: int,
            input_channels_lidar: int = 3,
            input_channels_sentinel2: int = 4,
            num_layers: int = 5,
            fusion_idx: int = 2,
            features_start: int = 64,
            upsampling: str = 'nearest',
            upsampling_sentinel2: str = "bilinear",
            act_fct: str = 'ELU',
            norm_layer: str = "bn",
            norm_layer_up: str = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels_lidar = input_channels_lidar
        self.input_channels_sentinel2 = input_channels_sentinel2
        self.fusion_idx = fusion_idx
        self.features_start = features_start
        self.upsampling = upsampling
        self.upsampling_sentinel2 = upsampling_sentinel2

        self.act_fct = act_fct
        self.act_fct_ = get_act(act_fct)

        self.norm_layer = norm_layer
        self.norm_layer_ = get_norm_layer(norm_layer)

        self.norm_layer_up = norm_layer_up
        if norm_layer_up is None:
            self.norm_layer_up_ = self.norm_layer_
        else:
            self.norm_layer_up_ = get_norm_layer(norm_layer_up)

        if num_layers < 1:
            raise ValueError(f'num_layers = {num_layers}, expected: num_layers > 0')

        self.num_layers = num_layers

        self.sentinel_upsample = lambda x, size: F.interpolate(
            x, size=size, mode=upsampling_sentinel2, align_corners=True
        )
        input_channels_sentinel2 = input_channels_sentinel2  # output of upscale model

        input_dim = input_channels_lidar
        if 0 == fusion_idx:
            input_dim += input_channels_sentinel2

        down_layers = [DoubleConv(input_dim, features_start, self.act_fct_, self.norm_layer_)]

        feats = features_start
        for i in range(1, num_layers):
            input_dim = feats
            feats *= 2
            if i == fusion_idx:
                input_dim += input_channels_sentinel2
            down_layers.append(Down(input_dim, feats, self.act_fct_, self.norm_layer_))
        self.down_layers = nn.ModuleList(down_layers)

        up_layers = []
        for _ in range(num_layers - 1):
            up_layers.append(Up(feats, feats // 2, upsampling, self.act_fct_, self.norm_layer_up_))
            feats //= 2
        self.up_layers = nn.ModuleList(up_layers)

        self.out = nn.Conv2d(feats, num_classes, kernel_size=1)

        self.idx_fusion = fusion_idx

    def forward(self, lidar, sentinel2):
        x = lidar

        shortcut = []
        # Down path
        for i, layer in enumerate(self.down_layers):
            if i == self.idx_fusion:
                sentinel2 = self.sentinel_upsample(sentinel2, x.shape[2:])
                x = torch.cat([x, sentinel2], dim=1)
            x = layer(x)
            if i < len(self.down_layers) - 1:
                shortcut.append(x)
        # Up path
        shortcut = shortcut[::-1]  # reverse

        for i, layer in enumerate(self.up_layers):
            x = layer(x, shortcut[i])
        return self.out(x)

    @classmethod
    def load_model(nn_model_cls, path, device):
        st_dict = torch.load(path, map_location=device)

        net = nn_model_cls(num_classes=st_dict['num_classes'],
                           input_channels_lidar=st_dict['input_channels_lidar'],
                           input_channels_sentinel2=st_dict['input_channels_sentinel2'],
                           num_layers=st_dict['num_layers'],
                           fusion_idx=st_dict['fusion_idx'],
                           features_start=st_dict['features_start'],
                           upsampling=st_dict['upsampling'],
                           upsampling_sentinel2=st_dict['upsampling_sentinel2'],
                           act_fct=st_dict['act_fct'],
                           norm_layer=st_dict['norm_layer'],
                           norm_layer_up=st_dict['norm_layer_up'],
                           )
        net.load_state_dict(st_dict['net_params'])
        net.to(device=device)
        return net

    @staticmethod
    def load_optimizer(path, optm, device):
        st_dict = torch.load(path, map_location=device)
        optm.load_state_dict(st_dict['optm_params'])
        return optm

    @staticmethod
    def save_model(model, path, optm=None):
        net = model
        if isinstance(model, nn.DataParallel):
            net = model.module

        st_dict = {'net_params': net.state_dict()}

        if optm is not None:
            st_dict['optm_params'] = optm.state_dict()

        st_dict['num_classes'] = net.num_classes
        st_dict['input_channels_lidar'] = net.input_channels_lidar
        st_dict['input_channels_sentinel2'] = net.input_channels_sentinel2
        st_dict['num_layers'] = net.num_layers
        st_dict['fusion_idx'] = net.fusion_idx
        st_dict['features_start'] = net.features_start
        st_dict['upsampling'] = net.upsampling
        st_dict['upsampling_sentinel2'] = net.upsampling_sentinel2
        st_dict['act_fct'] = net.act_fct
        st_dict['norm_layer'] = net.norm_layer
        st_dict['norm_layer_up'] = net.norm_layer_up
        try:
            torch.save(st_dict, path)
            return 1

        except Exception as e:
            return 0


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ELU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, act_fct: nn.Module, norm_layer: nn.Module):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), norm_layer(out_ch), act_fct,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), norm_layer(out_ch), act_fct
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int, act_fct: nn.Module, norm_layer: nn.Module):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch, act_fct, norm_layer))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, upsampling: str, act_fct: nn.Module, norm_layer: nn.Module):
        super().__init__()
        self.upsample = None
        if upsampling == 'upconf':
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        elif upsampling == 'nearest':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling, align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )

        self.conv = DoubleConv(in_ch, out_ch, act_fct, norm_layer)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

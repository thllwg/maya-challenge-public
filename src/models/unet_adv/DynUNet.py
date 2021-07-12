import torch
import torch.nn as nn
import torch.nn.functional as F

from .mnasnet_generator import mnasnet1_0, mnasnet1_3
from .resnet_generator import resnet101, resnet50, resnext50_32x4d, wide_resnet50_2, resnet18
from .standard_generator import StandardNet
from .swin_generator import swin_b, swin_s
from .util import init_weights, CustomPixelShuffle_ICNR, custom_conv_layer, noop, get_act, get_norm_layer


class UNet(nn.Module):

    def __init__(self, n_classes, n_features_lidar, n_features_sentinel2,
                 arch="mnasnet", imsize_lidar=(480, 480), imsize_sentinel2=(480 // 20, 480 // 20),
                 norm_type_encoder="bn", norm_type_decoder=None, upsampling="pixelshuffle",
                 blur_final=True, blur=True, nf_factor: float = 1.0, activation="elu", self_attention: bool = True,
                 backbone_warmup: int = 0):
        super().__init__()
        self.norm_type_encoder = norm_type_encoder
        self.norm_type_decoder = norm_type_encoder if norm_type_decoder is None else norm_type_decoder
        self.norm_layer_encoder = get_norm_layer(norm_type_encoder)
        self.norm_layer_decoder = get_norm_layer(norm_type_decoder)
        self.n_classes = n_classes
        self.n_features_lidar = n_features_lidar
        self.n_features_sentinel2 = n_features_sentinel2
        self.blur = blur
        self.blur_final = blur_final
        self.self_attention = self_attention
        self.nf_factor = nf_factor
        self.activation = activation
        self.act_fn = get_act(activation)
        self.imsize_lidar = imsize_lidar
        self.imsize_sentinel2 = imsize_sentinel2
        self.upsampling = upsampling
        self.arch = arch
        self.epoch = 0
        self.backbone_warmup = backbone_warmup

        self.arch_net = self.get_arch(arch)

        # first layer is stem, then blocks
        if arch in ["swin-b", "swin-s"]:
            self.encoder = self.arch_net()
            self.scale_factor = 4
        else:
            self.encoder = self.arch_net(
                c_in=self.n_features_lidar, act_fn=self.act_fn, norm_layer=self.norm_layer_encoder
            )
            self.scale_factor = 2

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # create dummy features to test network integrity
        lidar = torch.rand(2, n_features_lidar, *imsize_lidar).detach()
        sentinel2 = torch.rand(2, n_features_sentinel2, *imsize_sentinel2).detach()
        self.sentinel_upsample = lambda x, size: F.interpolate(
            x, size=size, mode="bilinear", align_corners=True
        )

        # iterate through encoder to dynamically measure sizes
        x = lidar

        outs = self.encoder(x)
        x = outs[-1]
        del outs[-1]

        in_sizes = [out.size(1) for out in outs]

        # invert sizes and outputs
        cross_sizes = in_sizes[::-1]
        outs = outs[::-1]

        # concat sentinel2
        x = torch.cat([x, self.sentinel_upsample(sentinel2, x.shape[2:])], 1)

        # init the cross_conv
        self.middle_conv = ResBlock(
            x.size(1), x.size(1) * 2, self.norm_type_encoder, self.norm_layer_encoder, self.act_fn
        ).eval()

        x = self.middle_conv(x)

        # init up scaling
        up_layers = []
        for i, cross_size in enumerate(cross_sizes):
            not_final = i != len(cross_sizes) - 1
            up_in_c = x.size(1)
            do_blur = blur and (not_final or blur_final)
            sa = (i == len(cross_sizes) - 3) and self_attention
            unet_block = UnetBlockDeep(up_in_c, cross_size, self.act_fn, self.norm_layer_decoder,
                                       final_div=not_final, blur=do_blur, self_attention=sa,
                                       norm_type=self.norm_type_decoder, nf_factor=nf_factor,
                                       upsampling=self.upsampling, scale_factor=self.scale_factor).eval()
            up_layers.append(unet_block)

            x = unet_block(x, outs[i])

        self.up_layers = nn.ModuleList(up_layers)
        last_dim = x.size(1)

        # init final output layers
        if lidar.size(2) == x.size(2) and lidar.size(3) == x.size(3):
            self.pre_final_conv = CatResBlock(
                last_dim + n_features_lidar, (last_dim + n_features_lidar) * 2,
                self.norm_type_decoder, self.norm_layer_decoder, self.act_fn
            ).eval()
        else:  # if a final upscaling is needed
            self.pre_final_conv = UnetBlockDeep(
                last_dim, n_features_lidar, self.act_fn, self.norm_layer_decoder,
                final_div=True, blur=blur and (not_final or blur_final), self_attention=False,
                norm_type=self.norm_type_decoder, nf_factor=nf_factor,
                upsampling=self.upsampling, scale_factor=self.scale_factor
            ).eval()
        x = self.pre_final_conv(x, lidar)

        self.last_conv = nn.Conv2d(x.size(1), self.n_classes, 1).eval()
        self.last_conv(x)
        self.reset_parameters()

    def forward(self, lidar, sentinel2):
        x = lidar
        outs = self.encoder(x)
        x = outs[-1]
        del outs[-1]

        outs = outs[::-1]
        x = torch.cat([x, self.sentinel_upsample(sentinel2, x.shape[2:])], 1)

        x = self.middle_conv(x)

        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, outs[i])
        out = self.last_conv(self.pre_final_conv(x, lidar))
        return out

    def reset_parameters(self):
        # TODO this ignores zero_bn settings
        if self.arch in ["swin-b", "swin-s"]:  # skip pretrained encoder
            self.up_layers.apply(init_weights)
            self.middle_conv.apply(init_weights)
            self.pre_final_conv.apply(init_weights)
            self.last_conv.apply(init_weights)
        else:  # reset everything
            self.apply(init_weights)

    @staticmethod
    def get_arch(arch):
        if isinstance(arch, nn.Module):
            return arch
        elif arch in ["mnasnet", "mnasnet10", "mnasnet1", "mnasnet_1_0"]:
            return mnasnet1_0
        elif arch in ["mnasnet13", "mnasnet_1_3"]:
            return mnasnet1_3
        elif arch == "standard":
            return StandardNet
        elif arch == "resnet18":
            return resnet18
        elif arch == "resnet50":
            return resnet50
        elif arch == "wideresnet50":
            return wide_resnet50_2
        elif arch == "resnext50":
            return resnext50_32x4d
        elif arch == "resnet101":
            return resnet101
        elif arch == "swin-b":
            return swin_b
        elif arch == "swin-s":
            return swin_s
        else:
            raise Exception(f"Architecture not found: {str(arch)}")

    def set_current_epoch(self, epoch):
        self.epoch = epoch
        if self.backbone_warmup <= epoch:
            for param in self.encoder.parameters():
                param.requires_grad = True

    @staticmethod
    def load_model(path, device):
        st_dict = torch.load(path, map_location=device)

        net = UNet(n_classes=st_dict['n_classes'],
                   n_features_lidar=st_dict['n_features_lidar'],
                   n_features_sentinel2=st_dict['n_features_sentinel2'],
                   upsampling=st_dict['upsampling'],
                   activation=st_dict['activation'],
                   norm_type_encoder=st_dict['norm_type_encoder'],
                   norm_type_decoder=st_dict['norm_type_decoder'],
                   self_attention=st_dict['self_attention'],
                   blur=st_dict['blur'],
                   blur_final=st_dict['blur_final'],
                   arch=st_dict['arch'],
                   nf_factor=st_dict['nf_factor'],
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

        st_dict['n_classes'] = net.n_classes
        st_dict['n_features_lidar'] = net.n_features_lidar
        st_dict['n_features_sentinel2'] = net.n_features_sentinel2
        st_dict['upsampling'] = net.upsampling
        st_dict['activation'] = net.activation
        st_dict['norm_type_encoder'] = net.norm_type_encoder
        st_dict['norm_type_decoder'] = net.norm_type_decoder
        st_dict['self_attention'] = net.self_attention
        st_dict['blur'] = net.blur
        st_dict['blur_final'] = net.blur_final
        st_dict['arch'] = net.arch
        st_dict['nf_factor'] = net.nf_factor

        try:
            torch.save(st_dict, path)
            return 1

        except Exception as e:
            return 0


class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self, up_in_c: int, x_in_c: int, act_fn: nn.Module, norm_layer: nn.Module, final_div: bool = True,
                 blur: bool = False, self_attention: bool = False, nf_factor: float = 1.0, norm_type="spectral",
                 upsampling="pixelshuffle", scale_factor=2, **kwargs):
        super().__init__()
        self.shuf = []
        up_in_shuf = up_in_c
        for i in range(scale_factor // 2):
            if upsampling == "pixelshuffle":
                self.shuf.append(
                    CustomPixelShuffle_ICNR(up_in_shuf, act_fn, up_in_shuf // 2, blur=blur, scale=2, **kwargs)
                )
            elif upsampling == "nearest":
                self.shuf.append(nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=upsampling),
                    custom_conv_layer(up_in_shuf, up_in_shuf // 2, act_fn=act_fn, norm_layer=norm_layer, **kwargs),
                ))
            else:
                self.shuf.append(nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=upsampling, align_corners=True),
                    custom_conv_layer(up_in_shuf, up_in_shuf // 2, act_fn=act_fn, norm_layer=norm_layer, **kwargs),
                ))
            up_in_shuf //= 2
        self.shuf = nn.Sequential(*self.shuf)

        ni = up_in_shuf + x_in_c
        nf = int((ni if final_div else up_in_shuf) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, act_fn=act_fn, norm_layer=norm_layer, norm_type=norm_type, **kwargs)
        self.conv2 = custom_conv_layer(
            nf, nf, act_fn=act_fn, norm_layer=norm_layer, norm_type=norm_type, self_attention=self_attention, **kwargs
        )
        self.idconv = noop if ni == nf else custom_conv_layer(
            ni, nf, 1, norm_layer=norm_layer, norm_type=norm_type, act_fn=None, **kwargs
        )
        self.relu = act_fn

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        ssh = skip.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, skip.shape[-2:], mode='nearest')

        cat_x = torch.cat([up_out, skip], dim=1)

        return self.conv2(self.conv1(self.relu(cat_x))) + self.idconv(cat_x)


class ResBlock(nn.Module):
    def __init__(self, ni, nh, norm_type, norm_layer, act_fn):
        super().__init__()
        self.convs = nn.Sequential(
            custom_conv_layer(ni, nh, 3, norm_type=norm_type, norm_layer=norm_layer, act_fn=act_fn),
            custom_conv_layer(nh, ni, 3, norm_type=norm_type, norm_layer=norm_layer)
        )

    def forward(self, x): return self.convs(x) + x


class CatResBlock(ResBlock):
    def forward(self, x, lidar):
        x = torch.cat([x, lidar], 1)
        return self.convs(x) + x

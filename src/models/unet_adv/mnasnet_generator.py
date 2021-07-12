# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MNASNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, act_fn, norm_layer):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            norm_layer(mid_ch),
            act_fn,
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            norm_layer(mid_ch),
            act_fn,
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            norm_layer(out_ch)
        )

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, act_fn, norm_layer):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, act_fn, norm_layer)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, act_fn, norm_layer)
        )
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rounding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha: float, act_fn: nn.Module, norm_layer: nn.Module, c_in: int):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        depths = _get_depths(alpha)
        layers = [
            nn.Sequential(
                # First layer: regular conv.
                nn.Conv2d(c_in, depths[0], 3, padding=1, stride=1, bias=False),
                norm_layer(depths[0]),
                act_fn,
                # Depthwise separable, no skip.
                nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False),
                norm_layer(depths[0]),
                act_fn,
                nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
                norm_layer(depths[1]),
            ),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, act_fn, norm_layer),
            _stack(depths[2], depths[3], 5, 2, 3, 3, act_fn, norm_layer),
            nn.Sequential(
                _stack(depths[3], depths[4], 5, 2, 6, 3, act_fn, norm_layer),
                _stack(depths[4], depths[5], 3, 1, 6, 2, act_fn, norm_layer),
            ),
            nn.Sequential(
                _stack(depths[5], depths[6], 5, 2, 6, 4, act_fn, norm_layer),
                _stack(depths[6], depths[7], 3, 1, 6, 1, act_fn, norm_layer),
            )
        ]

        self.blocks = nn.ModuleList(layers)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(MNASNet, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(F.dropout2d(x, .1, self.training))
            out.append(x)
        return out


def mnasnet0_5(**kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(0.5, **kwargs)
    return model


def mnasnet0_75(**kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(0.75, **kwargs)
    return model


def mnasnet1_0(**kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(1.0, **kwargs)
    return model


def mnasnet1_3(**kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(1.3, **kwargs)
    return model

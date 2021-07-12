import torch
import torch.nn as nn
import torch.nn.functional as F


def noop(x): return x


# or: ELU+init (a=0.54; gain=1.55)
# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def get_act(name):
    if name == "elu":
        return nn.ELU(inplace=True, alpha=0.54)
    elif name == "swish":
        return MemoryEfficientSwish()
    elif name == "mish":
        return Mish()
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
    elif name in ["no", "none"]:
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
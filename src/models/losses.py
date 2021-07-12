import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float, reduction="mean") -> None:
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-6

        if reduction == "mean":
            self.red = torch.mean
        elif reduction == "sum":
            self.red = torch.sum

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # compute sigmoid over the classes axis
        input = torch.sigmoid(input)

        # flatten label and prediction tensors
        input = input.view(-1)
        target = target.view(-1)

        intersection = torch.sum(input * target)
        fps = torch.sum(input * (1. - target))
        fns = torch.sum((1. - input) * target)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)
        return self.red((1. - tversky_loss) ** self.gamma)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_feat = vgg16_bn(True).features.eval()
        for p in self.m_feat.parameters(): p.requires_grad = False
        blocks = [i - 1 for i, o in enumerate(list(self.m_feat.children())) if isinstance(o, nn.MaxPool2d)]
        self.layer_ids = blocks[2:5]
        self.wgts = [20, 70, 10]
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(self.layer_ids))]
        self.first_loss = F.binary_cross_entropy_with_logits
        self.base_loss = F.smooth_l1_loss

    def _make_features(self, x, clone=False):
        prev_id = 0
        outs = []
        for layer_id in self.layer_ids:
            layer_id += 1
            x = self.m_feat[prev_id:layer_id](x)
            outs.append(x.clone() if clone else x)
            prev_id = layer_id
        return outs

    def forward(self, inputs: torch.Tensor, target: torch.Tensor):
        self.feat_losses = [100 * self.first_loss(inputs, target)]

        out_feat = self._make_features(torch.repeat_interleave(target, 3, 1), clone=True)
        in_feat = self._make_features(torch.repeat_interleave(inputs, 3, 1))

        self.feat_losses.extend(
            [self.base_loss(f_in, f_out) * w for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        )

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)


def iou(pred, target, absent_score=0.0):
    numerator = torch.sum(pred * target, dim=[1, 2, 3])  # TP
    denominator = torch.sum(pred + target, dim=[1, 2, 3]) - numerator  # 2TP + FP + FN - TP
    iou = (numerator) / (denominator)
    iou[denominator == 0] = absent_score
    return iou.mean()


def dice_coeff(pred, target):
    # F1 = TP / (TP + 0.5 (FP + FN)) = 2TP / (2TP + FP + FN)
    numerator = 2 * torch.sum(pred * target)  # 2TP
    denominator = torch.sum(pred + target)  # 2TP + FP + FN
    if (denominator == 0):
        return torch.tensor(0.).to(pred.device)
    return (numerator) / (denominator)


# Predictions are logits and the masks are 0-1
# Logits > 0. is same as sigmoid > 0.5

def div_values(dict1, scalar):
    dict1["building"] /= scalar
    dict1["platform"] /= scalar
    dict1["aguada"] /= scalar
    return dict1


def add_values(dict1, dict2):
    dict1["building"] += dict2["building"]
    dict1["platform"] += dict2["platform"]
    dict1["aguada"] += dict2["aguada"]
    return dict1


def maya_iou(pred, mask_building, mask_platform, mask_aguada, absent_score=0.0):
    pred = (pred > 0.).float()
    return {'building': iou(pred[:, [0]], mask_building, absent_score).item(),
            'platform': iou(pred[:, [1]], mask_platform, absent_score).item(),
            'aguada': iou(pred[:, [2]], mask_aguada, absent_score).item()}


def maya_dice_coeff(pred, mask_building, mask_platform, mask_aguada):
    pred = (pred > 0.).float()
    return {'building': dice_coeff(pred[:, [0]], mask_building).item(),
            'platform': dice_coeff(pred[:, [1]], mask_platform).item(),
            'aguada': dice_coeff(pred[:, [2]], mask_aguada).item()}


def add_to_dataset_loss(ls, part):
    ls['building'] += part['building']
    ls['platform'] += part['platform']
    ls['aguada'] += part['aguada']
    return ls


def normalize_loss(ls, n_val):
    ls['building'] /= n_val
    ls['platform'] /= n_val
    ls['aguada'] /= n_val
    return ls

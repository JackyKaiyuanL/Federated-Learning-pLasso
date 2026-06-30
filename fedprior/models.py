"""Models and the client container for the FedPrior experiments.

Two tasks are supported:

* ``linear`` – the synthetic softmax benchmark (60 features + 1 constant = 61
  inputs, 10 classes). Bias is frozen at 0 and carried by the constant feature,
  matching the original notebooks.
* ``cnn``    – a small CNN for MNIST (1x28x28 -> 10), used for the non-IID
  benchmark where client-drift effects actually appear (a non-convex model).

Everything downstream is model-agnostic: aggregation averages full
``state_dict``s, so any ``nn.Module`` works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

NUM_INPUTS = 61   # synthetic: 60 features + 1 appended constant (absorbs bias)
NUM_CLASSES = 10


class LinearNet(nn.Module):
    """Softmax classifier = one linear layer. Bias frozen at 0."""

    def __init__(self, num_inputs: int = NUM_INPUTS, num_outputs: int = NUM_CLASSES):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.constant_(self.linear.bias, 0.0)
        self.linear.bias.requires_grad_(False)   # bias carried by the constant feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.shape[0], -1))


class SmallCNN(nn.Module):
    """LeNet-style CNN for MNIST (1x28x28 -> 10)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                   # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:                       # (N, 784) -> (N,1,28,28)
            x = x.view(-1, 1, 28, 28)
        return self.classifier(self.features(x))


def _gn(channels, groups=2):
    # GroupNorm is the FL-safe normalization: unlike BatchNorm its statistics
    # are parameter-free per sample, so averaging client models is well-defined.
    return nn.GroupNorm(min(groups, channels), channels)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.n1 = _gn(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.n2 = _gn(out_c)
        self.short = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), _gn(out_c))

    def forward(self, x):
        out = torch.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return torch.relu(out + self.short(x))


class ResNetGN(nn.Module):
    """ResNet-18 with GroupNorm. ``stem='cifar'`` (3x3 s1, for 32x32) or
    ``stem='imagenet'`` (7x7 s2 + maxpool, for larger images)."""

    def __init__(self, num_classes=10, stem="cifar", blocks=(2, 2, 2, 2),
                 widths=(64, 128, 256, 512)):
        super().__init__()
        if stem == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), _gn(widths[0]),
                nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, widths[0], 7, 2, 3, bias=False), _gn(widths[0]),
                nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.in_c = widths[0]
        layers = []
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            layers.append(self._make(w, n, stride))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], num_classes)

    def _make(self, out_c, n, stride):
        blk, strides = [], [stride] + [1] * (n - 1)
        for s in strides:
            blk.append(_BasicBlock(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.head(x)


class PretrainedResNetML(nn.Module):
    """ImageNet-pretrained ResNet-18 with BatchNorm FROZEN (eval mode, no grad).

    Frozen BN makes this FL-safe: the only trainable/aggregated parameters are the
    conv and linear weights, while BN affine params and running statistics stay at
    their ImageNet values on every client, so model averaging is well-defined (the
    same reason we use GroupNorm in the from-scratch model). Inputs are
    ImageNet-normalized in ``data.py``.
    """

    def __init__(self, num_classes=20):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        self._freeze_bn()

    def _freeze_bn(self):
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_bn()                 # keep BN in eval even during training
        return self

    def forward(self, x):
        return self.net(x)


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name == "linear":
        return LinearNet(**kw)
    if name == "cnn":
        return SmallCNN(**kw)
    if name == "resnet_ml_pt":     # ImageNet-pretrained ResNet-18, frozen BN (multi-label)
        kw.setdefault("num_classes", 20)
        return PretrainedResNetML(**kw)
    if name == "resnet":          # CIFAR-10 (stem='cifar', num_classes=10)
        return ResNetGN(stem="cifar", **kw)
    if name == "resnet_ml":       # multi-label (stem='imagenet', num_classes=20)
        kw.setdefault("num_classes", 20)
        return ResNetGN(stem="imagenet", **kw)
    raise ValueError(f"unknown model '{name}'")


@dataclass
class Client:
    """A federated client holding its local (already tensorised) data."""

    x: torch.Tensor
    y: torch.Tensor
    w_cur: object = field(default=None)        # last local state_dict (set by driver)

    @property
    def number(self) -> int:
        return len(self.y)

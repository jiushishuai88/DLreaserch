import torch.nn as nn
import math
import torch

__all__ = ['densebc_100_12']


class BottleNeck(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channel, 4 * growth_rate, kernel_size=1, stride=1, padding=0,
                                             bias=False))

        self.conv2 = nn.Sequential(nn.BatchNorm2d(4 * growth_rate),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1,
                                             bias=False))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.cat([out, x], 1)
        return out


class DenseNet(nn.Module):
    def __init__(self, depth, grow_rate, num_classes, reduction=0.5):
        super(DenseNet, self).__init__()
        self.grow_rate = grow_rate

        num_planes = 2 * grow_rate
        num_blocks = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layers(num_planes, num_blocks)
        num_planes = num_planes + grow_rate * num_blocks
        num_out = math.floor(num_planes * reduction)
        self.transition1 = self._transition(num_planes, num_out)
        num_planes = num_out

        self.block2 = self._make_layers(num_planes, num_blocks)
        num_planes = num_planes + grow_rate * num_blocks
        num_out = math.floor(num_planes * reduction)
        self.transition2 = self._transition(num_planes, num_out)
        num_planes = num_out

        self.block3 = self._make_layers(num_planes, num_blocks)
        num_planes = num_planes + grow_rate * num_blocks

        self.transition3 = nn.Sequential(nn.BatchNorm2d(num_planes),
                                nn.ReLU(inplace=True),
                                nn.AvgPool2d(kernel_size=8, stride=1))
        self.fc = nn.Linear(num_planes,num_classes)

    # bn,relu,conv,avgpool
    def _transition(self, in_channel, out_channel):
        return nn.Sequential(nn.BatchNorm2d(in_channel),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.AvgPool2d(2, stride=2))

    def _make_layers(self, in_channel, num_blocks):
        layers = []
        channel = in_channel
        for i in range(num_blocks):
            layers.append(BottleNeck(channel, self.grow_rate))
            channel += self.grow_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.transition1(out)
        out = self.block2(out)
        out = self.transition2(out)
        out = self.block3(out)
        out = self.transition3(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


def densebc_100_12(num_classes):
    return DenseNet(100, 12, num_classes, 0.5)

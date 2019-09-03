import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['sk_preresneXt29_32x8d']


class SKConv(nn.Module):
    def __init__(self, channel, M, cardinality):
        super(SKConv, self).__init__()
        d = max(channel // 32, 32)
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, dilation=1 + i, stride=1, padding=1 + i,groups=cardinality),
                              nn.BatchNorm2d(channel),
                              nn.ReLU(inplace=True)
                              ))

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, channel))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            temp = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = temp
            else:
                feas = torch.cat([feas, temp], dim=1)
        U = torch.sum(feas, dim=1)
        S = self.globalpool(U).squeeze_()
        Z = self.fc(S)
        for i, fc in enumerate(self.fcs):
            tempa = fc(Z).unsqueeze_(dim=1)
            if i == 0:
                a = tempa
            else:
                a = torch.cat([a, tempa], dim=1)
        A = self.softmax(a).unsqueeze_(dim=-1).unsqueeze_(dim=-1)
        V = torch.sum(feas * A, dim=1)
        return V


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, cardinality=8, base_width=64, index=1, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        D = cardinality * base_width * index
        self.bn_1 = nn.BatchNorm2d(in_channel)
        self.conv_1 = nn.Conv2d(in_channel, D, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn_2 = nn.BatchNorm2d(D)
        self.conv_2 = SKConv(D, 2, cardinality)

        self.bn_3 = nn.BatchNorm2d(D)
        self.conv_3 = nn.Conv2d(D, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn_1(x)
        out = self.relu(out)
        out = self.conv_1(out)

        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_2(out)

        out = self.bn_3(out)
        out = self.relu(out)
        out = self.conv_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual


class SKPreResneXt(nn.Module):
    def __init__(self, depth, num_classes=100, cardinality=8, base_width=64):
        super(SKPreResneXt, self).__init__()
        assert (depth - 2) % 9 == 0, "When use bottleneck, depth 9n+2"
        n = (depth - 2) // 9
        block = BottleNeck
        self.cardinality = cardinality
        self.base_width = base_width

        self.in_channels = 64
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 256, n, index=1)
        self.layer2 = self._make_layer(block, 512, n, index=2, stride=2)
        self.layer3 = self._make_layer(block, 1024, n, index=3, stride=2)
        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, index=1, stride=1):  # 16 10
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False))
        layers = [block(self.in_channels, channels, self.cardinality, self.base_width, index=index, stride=stride,
                        downsample=downsample)]

        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(channels, channels, self.cardinality, self.base_width, index=index, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)  # out 32x32x16

        x = self.layer1(x)  # out 32x32x16
        x = self.layer2(x)  # out 16x16x32
        x = self.layer3(x)  # out 8x8x64

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)  # out 4x4x64
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def sk_preresneXt29_32x8d(num_class):
    return SKPreResneXt(depth=29, num_classes=num_class, cardinality=32, base_width=8)

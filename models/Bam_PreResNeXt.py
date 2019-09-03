import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['bam_preresneXt29_32x8d']


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        out = self.compress(x)
        out = self.conv(out)
        out = torch.sigmoid(out)
        return x * out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels):
        super(ChannelGate,self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // 16),
                                 nn.ReLU(inplace=True), nn.Linear(gate_channels // 16, gate_channels))

    def forward(self, x):
        mavg = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        mavg = self.mlp(mavg)
        mmax = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        mmax = self.mlp(mmax)
        mf = mavg + mmax
        mf = torch.sigmoid(mf).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * mf


class CBAM(nn.Module):
    def __init__(self, gate_channels):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        out = self.channel_gate(x)
        out = self.spatial_gate(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, cardinality=8, base_width=64, index=1, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        D = cardinality * base_width * index
        self.bn_1 = nn.BatchNorm2d(in_channel)
        self.conv_1 = nn.Conv2d(in_channel, D, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn_2 = nn.BatchNorm2d(D)
        self.conv_2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)

        self.bn_3 = nn.BatchNorm2d(D)
        self.conv_3 = nn.Conv2d(D, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.CBAM = CBAM(out_channel)

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

        out = self.CBAM(out)

        return out + residual


class BamPreResneXt(nn.Module):
    def __init__(self, depth, num_classes=100, cardinality=8, base_width=64):
        super(BamPreResneXt, self).__init__()
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


def bam_preresneXt29_32x8d(num_class):
    return BamPreResneXt(depth=29, num_classes=num_class, cardinality=32, base_width=8)

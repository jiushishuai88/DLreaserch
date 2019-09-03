import torch.nn as nn

__all__ = ['preresneXt29_8x64d', 'preresneXt29_16x64d', 'preresneXt29_32x32d', 'preresneXt29_32x16d',
           'preresneXt29_32x8d']


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
        self.stride = stride

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


class PreResneXt(nn.Module):
    def __init__(self, depth, num_classes=100, cardinality=8, base_width=64):
        super(PreResneXt, self).__init__()
        assert (depth - 2) % 9 == 0, "When use bottleneck, depth 12n+2"
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


def preresneXt29_8x64d(num_class):
    return PreResneXt(depth=29, num_classes=num_class, cardinality=8, base_width=64)


def preresneXt29_16x64d(num_class):
    return PreResneXt(depth=29, num_classes=num_class, cardinality=16, base_width=64)


def preresneXt29_32x32d(num_class):
    return PreResneXt(depth=29, num_classes=num_class, cardinality=32, base_width=32)


def preresneXt29_32x16d(num_class):
    return PreResneXt(depth=29, num_classes=num_class, cardinality=32, base_width=16)


def preresneXt29_32x8d(num_class):
    return PreResneXt(depth=29, num_classes=num_class, cardinality=32, base_width=8)

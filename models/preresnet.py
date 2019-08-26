import torch.nn as nn

__all__ = ['preresnet26', 'preresnet74', 'preresnet146','preresnet1094']


def conv3by3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv3by3(inplanes, planes, stride)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_2 = conv3by3(planes, planes)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        quarter_outchannel = out_channel // 4
        self.bn_1 = nn.BatchNorm2d(in_channel)
        self.conv_1 = nn.Conv2d(in_channel, quarter_outchannel, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn_2 = nn.BatchNorm2d(quarter_outchannel)
        self.conv_2 = nn.Conv2d(quarter_outchannel, quarter_outchannel, kernel_size=3, stride=stride, padding=1,
                                bias=False)

        self.bn_3 = nn.BatchNorm2d(quarter_outchannel)
        self.conv_3 = nn.Conv2d(quarter_outchannel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

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


class PreResnet(nn.Module):
    def __init__(self, depth, num_classes=100, is_basic=True):
        super(PreResnet, self).__init__()
        if is_basic:
            assert (depth - 2) % 8 == 0, "When use basicblock, depth 8n+2"
            n = (depth - 2) // 8
            block = BasicBlock
        else:
            assert (depth - 2) % 12 == 0, "When use bottleneck, depth 12n+2"
            n = (depth - 2) // 12
            block = BottleNeck

        self.in_channels = 64
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, n)
        self.layer2 = self._make_layer(block, 128, n, stride=2)
        self.layer3 = self._make_layer(block, 256, n, stride=2)
        self.layer4 = self._make_layer(block, 512, n, stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(2 * 2 * 512, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):  # 16 10
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels,channels, kernel_size=1, stride=stride, bias=False))
        layers = [block(self.in_channels, channels, stride, downsample=downsample)]

        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)  # out 32x32x16

        x = self.layer1(x)  # out 32x32x16
        x = self.layer2(x)  # out 16x16x32
        x = self.layer3(x)  # out 8x8x64
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)  # out 4x4x64
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preresnet26(num_class):
    return PreResnet(depth=26, num_classes=num_class,is_basic=False)


def preresnet74(num_class):
    return PreResnet(depth=74, num_classes=num_class,is_basic=False)


def preresnet146(num_class):
    return PreResnet(depth=146, num_classes=num_class,is_basic=False)

def preresnet1094(num_class):
    return PreResnet(depth=1094, num_classes=num_class,is_basic=False)

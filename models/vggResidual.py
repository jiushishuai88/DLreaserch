import torch.nn as nn

__all__ = ['vgg19Residual']
v19 = [[64, 2], 'max', [128, 2], 'max', [256, 4], 'max', [512, 4], 'max', [512, 4], 'max']


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, nums):
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   )
        self.relu = nn.ReLU(inplace=True)
        self.nums = nums

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        for i in range(self.nums):
            if 0 < i < (self.nums - 1):
                out = self.conv2(out)
            elif i == (self.nums - 1):
                out = self.conv3(out)
        return self.relu(residual + out)


class VggNet(nn.Module):
    def __init__(self, ops, num_class=10):
        super(VggNet, self).__init__()
        self.conv = ops
        self.fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        self.fc3 = nn.Linear(512, num_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def sequential_ops(cfg):
    ops = []
    last_channels = 3
    for v in cfg:
        if v == 'max':
            ops += [nn.MaxPool2d(2, stride=2)]
        else:
            ops += [Block(last_channels, v[0], v[1])]
            last_channels = v[0]
    return nn.Sequential(*ops)


def vgg19Residual(num_classes):
    return VggNet(sequential_ops(v19), num_classes)

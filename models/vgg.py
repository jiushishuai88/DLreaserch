import torch.nn as nn

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
v11 = [64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max']
v13 = [64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max']
v16 = [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512, 512, 512, 'max']
v19 = [64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512, 512, 512, 512, 'max']


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
            ops += [nn.Conv2d(last_channels, v, kernel_size=3, stride=1, padding=1)]
            ops += [nn.BatchNorm2d(v)]
            ops += [nn.ReLU(inplace=True)]
            last_channels = v
    return nn.Sequential(*ops)


def vgg11(num_classes):
    return VggNet(sequential_ops(v11), num_classes)


def vgg13(num_classes):
    return VggNet(sequential_ops(v13), num_classes)


def vgg16(num_classes):
    return VggNet(sequential_ops(v16), num_classes)


def vgg19(num_classes):
    return VggNet(sequential_ops(v19), num_classes)

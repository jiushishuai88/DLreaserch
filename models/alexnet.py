import torch.nn as nn

__all__ = ['alexnet']


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, (8, 8), stride=4, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, (5, 5), stride=1, padding=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, (3, 3), stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, stride=2)
                                   )
        self.fc6 = nn.Sequential(nn.Linear(256, 2048),
                                 nn.Dropout(p=0.2))
        self.fc7 = nn.Sequential(nn.Linear(2048, 2048),
                                 nn.Dropout(p=0.2))
        self.fc8 = nn.Linear(2048, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0),-1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return  x
def alexnet(classes):
    return  AlexNet(num_classes = classes)

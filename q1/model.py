import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 96, 7, stride=1, padding=3)
        self.local_response_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.max_pool1 = nn.MaxPool2d((3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d((3,3), stride=2, padding=0)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(512, 256, 7, stride=1, padding=3)
        self.conv7 = nn.Conv2d(256, 128, 11, stride=1, padding=5)
        self.conv8 = nn.Conv2d(128, 32, 11, stride=1, padding=5)
        # self.conv9 = nn.Conv2d(32, 1, 13, stride=1, padding=6)
        # I remove padding 2 from Conv2d Transpose because the output dimension was not right
        self.deconv = nn.ConvTranspose2d(32, 1, 8, stride=4, bias=False)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.local_response_norm(x)
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.deconv(x)
        return x


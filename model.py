import torch
import torch.nn as nn


class Conv2dBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResBlock, self).__init__()

        self.net1 = nn.Sequential(
            Conv2dBN(c_in=c_in, c_out=c_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.net2 = nn.Sequential(
            Conv2dBN(c_in=c_in, c_out=c_out, kernel_size=1, stride=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        return torch.add(self.net1(x), self.net2(x))


class ResPath(nn.Module):
    def __init__(self, c_in, c_out, lenght):
        super(ResPath, self).__init__()
        self.length = lenght
        self.c_in = c_in
        self.c_out = c_out
        self.net = self.__build_net()

    def forward(self, x):
        return self.net(x)

    def __build_net(self):
        net = [ResBlock(c_in=self.c_in, c_out=self.c_out)]
        for _ in range(self.length - 1):
            net.append((ResBlock(c_in=self.c_out, c_out=self.c_out)))
        return nn.Sequential(*net)


class DC_MiniBlock(nn.Module):
    def __init__(self, c_in, w):
        super(DC_MiniBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2dBN(c_in, int(w/6), kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            Conv2dBN(int(w/6), int(w/3), kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            Conv2dBN(int(w/3), int(w/2), kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        return torch.concatenate([h1, h2, h3])


class DC_Block(nn.Module):
    def __init__(self, c_in, w):
        super(DC_Block, self).__init__()
        self.left_net = DC_MiniBlock(c_in=c_in, w=w)
        self.right_net = DC_MiniBlock(c_in=c_in, w=w)

    def forward(self, x):
        return torch.add(self.left_net(x), self.right_net(x))






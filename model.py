import torch
import torch.nn as nn

w = [32, 64, 128, 256, 512]
res_dim = [32, 64, 128, 256]

class Conv2dBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, use_activation=True):
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding),
            nn.BatchNorm2d(c_out)
        )
        self.use_activation = use_activation
    def forward(self, x):
        h = nn.ReLU(self.net(x)) if self.use_activation else self.net(x)
        return h



class TransConv2DBN(nn.Module):
    def __init__(self, w_in, kernel_size, stride, padding, alpha=1.67):
        c_out = w_in
        w_in = w_in * alpha
        c_in = int(w_in/6) + int(w_in/3) + int(w_in/2)
        self.trans = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),

    def forward(self, x):
        return self.trans(x)


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
    def __init__(self, w_in, lenght, alpha = 1.67):
        super(ResPath, self).__init__()
        self.length = lenght
        self.c_in = int((w_in * alpha) / 6) + int((w_in * alpha) / 3) + int((w_in * alpha) / 2)
        self.c_out = w_in
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


class DC_Block_Encoder(nn.Module):
    def __init__(self, w_in, w, init_block=False, alpha=1.67, initial_channels=None):
        super(DC_Block_Encoder, self).__init__()
        w = w * alpha
        if init_block:
            c_in = initial_channels
        else:
            w_in = w_in * alpha
            c_in = int(w_in/6) + int(w_in/3) + int(w_in/2) + res_dim
        self.left_net = DC_MiniBlock(c_in=c_in, w=w)
        self.right_net = DC_MiniBlock(c_in=c_in, w=w)

    def forward(self, x):
        return torch.add(self.left_net(x), self.right_net(x))


class DC_Block_Decoder(nn.Module):
    def __init__(self, w_in, w, res_channels, alpha=1.67):
        super(DC_Block_Encoder, self).__init__()
        w = w * alpha
        c_in = w_in + res_channels
        self.left_net = DC_MiniBlock(c_in=c_in, w=w)
        self.right_net = DC_MiniBlock(c_in=c_in, w=w)

    def forward(self, x):
        return torch.add(self.left_net(x), self.right_net(x))


class DC_Unet(nn.Module):
    def __init__(self, initial_channels, num_classes):
        # Encoder layers
        super(DC_Unet, self).__init__()
        self.dc_block1 = DC_Block_Encoder(0, w[0], init_block=True, initial_channels=initial_channels)
        self.dc_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DC_Block_Encoder(w[0], w[1]))
        self.dc_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DC_Block_Encoder(w[1], w[2]))
        self.dc_block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DC_Block_Encoder(w[2], w[3]))
        self.dc_block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DC_Block_Encoder(w[3], w[4]),
            TransConv2DBN(w[4], kernel_size=2, stride=2, padding=0))

        # Decoder Layers
        self.dc_block6 = nn.Sequential(
            DC_Block_Decoder(w[4], w[3], res_channels=res_dim[3]),
            TransConv2DBN(w[3], kernel_size=2, stride=2, padding=0))

        self.dc_block7 = nn.Sequential(
            DC_Block_Decoder(w[3], w[2], res_channels=res_dim[2]),
            TransConv2DBN(w[2], kernel_size=2, stride=2, padding=0))

        self.dc_block8 = nn.Sequential(
            DC_Block_Decoder(w[2], w[1], res_channels=res_dim[1]),
            TransConv2DBN(w[1], kernel_size=2, stride=2, padding=0))

        self.dc_block9 = nn.Sequential(
            DC_Block_Decoder(w[1], w[0], res_channels=res_dim[0]))

        # Res Path
        self.res1 = ResPath(res_dim[0], 4)
        self.res2 = ResPath(res_dim[1], 3)
        self.res3 = ResPath(res_dim[2], 2)
        self.res4 = ResPath(res_dim[3], 1)

        # Final layer
        self.final_layer = nn.Sequential(Conv2dBN(c_in=int(w[0]/6) + int(w[0]/3) + int(w[0]/2),
                                                  c_out=num_classes,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  use_activation=False),
                                         nn.Sigmoid())

    def forward(self, x):
        # Encoder
        h1 = self.dc_block1(x)
        res_h1 = self.res1(h1)

        h2 = self.dc_block2(h1)
        res_h2 = self.res2(h2)

        h3 = self.dc_block2(h2)
        res_h3 = self.res2(h3)

        h4 = self.dc_block2(h3)
        res_h4 = self.res2(h4)

        h = self.dc_block5(h4)

        # Decoder
        h = torch.cat((h, res_h4))

        h = self.dc_block6(h)
        h = torch.cat(h, res_h3)

        h = self.dc_block7(h)
        h = torch.cat(h, res_h2)

        h = self.dc_block8(h)
        h = torch.cat(h, res_h1)

        h = self.dc_block9(h)

        # Output
        return self.final_layer(h)




import math

import torch
import torch.nn as nn
from thop import profile

from CSAM import CSAM
from RFE import PCRN


class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()

        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv3d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv3d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv3d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    # 前向传播方法，定义数据如何通过网络
    def forward(self, x):
        x = self.act(self.conv_0(x))

        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)

        x1 = self.act(self.conv_1(x1))


        x = self.conv_2(torch.cat([x1, x2], dim=1))


        return x

def channel_shuffle_3d(x, groups):
    batch, num_chan, depth, height, width = x.size()
    channel_per_group = num_chan // groups

    x = x.view(batch, groups, channel_per_group, depth, height, width)

    x = x.transpose(1, 2).contiguous()

    x = x.view(batch, -1, depth, height, width)
    return x

class RRAF(nn.Module):
    def __init__(self, in_channels=64):
        super(RRAF, self).__init__()
        self.rlfe1 = PCRN(in_channels)
        self.rlfe2 = PCRN(in_channels)
        self.rlfe3 = PCRN(in_channels)
        self.rlfe4 = PCRN(in_channels)
        self.rlfe5 = PCRN(in_channels)
        self.conv1 = nn.Conv3d(in_channels=64 * 2, out_channels=64, kernel_size=1)
        self.csam=CSAM(in_channels)
    def forward(self, x):
        e1 = self.rlfe1(x)
        e2 = self.rlfe2(e1) + e1
        e3 = self.rlfe3(e2) + e2
        e4 = self.rlfe4(e3)+e3
        e5 = self.rlfe5(e4)
        c1 = torch.cat([e4, e5], dim=1)
        f1 = self.conv1(channel_shuffle_3d(c1, 2))
        c2 = torch.cat([f1, e3], dim=1)
        f2 = self.conv1(channel_shuffle_3d(c2, 2))
        c3 = torch.cat([f2, e2], dim=1)
        f3 = self.conv1(channel_shuffle_3d(c3, 2))
        c4 = torch.cat([f3, e1], dim=1)
        f4 = self.conv1(channel_shuffle_3d(c4, 2))

        output = f4 + x
        output=self.csam(output)
        return output


class PixelShuffle3D(nn.Module):
    def __init__(self, up_scale):
        super(PixelShuffle3D, self).__init__()
        self.up_scale = up_scale  # 上采样因子

    def forward(self, x):

        batch_size, channels, D, H, W = x.shape

        assert channels % (self.up_scale ** 3) == 0, "The number of channels must be divisible by up_scale^3."

        out_channels = channels // (self.up_scale ** 3)

        x = x.view(batch_size, out_channels, self.up_scale, self.up_scale, self.up_scale, D, H, W)

        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        x = x.view(batch_size, out_channels, D * self.up_scale, H * self.up_scale, W * self.up_scale)
        return x


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels * up_scale ** 3, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle3D(up_scale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pixel_shuffle(x)
        return x



class RFF(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=2):

        super(RFF, self).__init__()
        self.channels = 64
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=self.channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.rraf1 = RRAF(in_channels=self.channels)
        self.rraf2 = RRAF(in_channels=self.channels)
        self.rraf3 = RRAF(in_channels=self.channels)


        pix_shuffle = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                pix_shuffle.append(UpsampleBLock(64, 2))
        self.pix_shuffle = nn.Sequential(*pix_shuffle)
        self.pcfn=PCFN(self.channels)
        self.finalconv = nn.Conv3d(self.channels, in_channels, kernel_size=3, padding=1, bias=True)
    def forward(self, x):
        e1 = self.relu(self.conv1(x))
        r1 = self.rraf1(e1)
        r2 = self.rraf2(r1)
        r3 = self.rraf3(r2)

        e2=self.pcfn(r3)
        out1 = self.pix_shuffle(e2)
        out2 = self.pix_shuffle(e1)
        output = self.finalconv(out1 + out2)

        return output


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myput = torch.zeros((1, 1, 10, 10, 10)).to(device)
    model = RFF(1,2).to(device)

    print(model(myput).shape)

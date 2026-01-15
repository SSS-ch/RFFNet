import math

import torch
import torch.nn as nn
from thop import profile


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))

        return out * x
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ChannelAttention, self).__init__()
        # 计算卷积核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1,1)

        return out * x

class CSAM(nn.Module):
    def __init__(self, in_channel):
        super(CSAM, self).__init__()
        self.eca=ChannelAttention(in_channel)
        self.st=SpatialAttention()
    def forward(self, x):
        e1=self.eca(x)

        e2=self.st(x)

        return e2+e1


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myput = torch.zeros((1, 64, 10, 10, 10)).to(device)
    model = CSAM(64).to(device)

    print(model(myput).shape)
    flops, params = profile(model, inputs=(myput,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))
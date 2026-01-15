import torch
import torch.nn as nn
from thop import profile


class RFE(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(RFE, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True)
    def forward(self, x):
        e1 = self.block1(x)
        e2 = self.conv1(e1 + x)
        output = e2
        return output

class PCRN(nn.Module):
    def __init__(self, channels):
        super(PCRN, self).__init__()
        self.residual = RFE(channels, 32)

    def forward(self, x):
        output = self.residual(x)
        output = x + output
        return output



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU，否则在CPU上训练
    myput = torch.zeros((1, 64, 10, 10, 10)).to(device)
    model = PCRN(64).to(device)

    print(model(myput).shape)
    flops, params = profile(model, inputs=(myput,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))
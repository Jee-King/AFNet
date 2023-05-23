import torch,os
import torch.nn as nn
from torch.nn.parameter import Parameter
from thop import profile
from thop import clever_format


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

if __name__ == '__main__':
    net = Fire(256,256,16)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = net.cuda()

    var1 = torch.FloatTensor(1, 256, 18, 18).cuda()

    macs, params = profile(net, inputs=(var1,))
    # out1  = net(var1)

    print(macs, params)


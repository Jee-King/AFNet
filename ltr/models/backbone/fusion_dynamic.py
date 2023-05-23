import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from thop import profile



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DCM(nn.Module):
    def __init__(self, filter_size, in_channels, channels):
        super(DCM, self).__init__()
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.channels = channels
        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.c1 = nn.Sequential( nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(self.in_channels),
                                 nn.ReLU() )

    def forward(self, x, convoluted):  # x for cal kernel, input is convoluted
        """Forward function."""
        generated_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        convoluted = self.c1(convoluted)
        b, c, h, w = x.shape
        # [1, b * c, h, w], c = self.channels
        convoluted = convoluted.view(1, b * c, h, w)
        # [b * c, 1, filter_size, filter_size]
        generated_filter = generated_filter.view(b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        convoluted = F.pad(input=convoluted, pad=p2d, mode='constant', value=0)
        # [1, b * c, h, w]
        output = F.conv2d(input=convoluted, weight=generated_filter, groups=b * c)
        # [b, c, h, w]
        output = output.view(b, c, h, w)

        return output


class Fusion_dynamic(nn.Module):
    def __init__(self, n_feat, kernel_size=3, padding=1, filter_ks=3, bias=False):
        super(Fusion_dynamic, self).__init__()
        self.trans = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=bias),
                                   nn.ReLU(),
                                   nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
        self.trans2 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=bias),
                                   nn.ReLU())
        self.F_mlp = nn.Sequential(nn.Linear(n_feat, 2*n_feat), nn.ReLU(), nn.Linear(2*n_feat, n_feat), nn.Sigmoid())
        ## dynamic filtering
        self.kernel_size = filter_ks
        self.dcm_e = DCM(self.kernel_size, in_channels=n_feat, channels=n_feat)
        self.dcm_f = DCM(self.kernel_size, in_channels=n_feat, channels=n_feat)
        self.gate_rgb = nn.Conv2d(n_feat, 1, kernel_size=1, bias=True)
        self.compress = ChannelPool()
        self.spatial_e = BasicConv(2, 1, 5, stride=1, padding=2, relu=False)
        self.conv1x1_fusion = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)


    def forward(self, event, frame):
        f3 =  self.dcm_f(event, frame)
        e3 = self.dcm_e(frame, event)
        f =  f3
        e =  e3
        res = self.conv1x1_fusion(torch.cat((e, f), dim=1))
        return res

if __name__ == '__main__':
    net = Fusion_dynamic(n_feat=256)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = net.cuda()
    var1 = torch.FloatTensor(1, 256, 18, 18).cuda()
    var2 = torch.FloatTensor(1, 256, 18, 18).cuda()
    macs, params = profile(net, inputs=(var1,var2))
    print(macs, params)

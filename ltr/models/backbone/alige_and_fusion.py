import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import sys

from basicsr.models.archs.arch_util import DCNv2Pack
from .adain import adaptive_instance_normalization as adain
from .EguideF import  EgF
from .fusion import  Fusion
from .fusion_dynamic import Fusion_dynamic


class DK(nn.Module):
    def __init__(self, in_channels, channels, filter_size):
        super(DK, self).__init__()
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_f = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.kernel_e = nn.Conv2d(self.in_channels, self.channels, 1, 1, 0)
        self.redu_f = nn.Sequential(nn.Conv2d(self.in_channels, self.channels, 1, 1, 0),
                                             nn.BatchNorm2d(self.channels),
                                             nn.ReLU())
        self.redu_e = nn.Sequential(nn.Conv2d(self.in_channels, self.channels, 1, 1, 0),
                                             nn.BatchNorm2d(self.channels),
                                             nn.ReLU())

        self.bottle_f = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, 1, 0),
                                             nn.BatchNorm2d(self.channels),
                                             nn.ReLU())
        self.bottle_e = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, 1, 0),
                                      nn.BatchNorm2d(self.channels),
                                      nn.ReLU())
    def forward(self, xf, xe):
        ker_f = self.kernel_f(F.adaptive_avg_pool2d(xf, self.filter_size))
        ker_e = self.kernel_e(F.adaptive_avg_pool2d(xe, self.filter_size))
        xf = self.redu_f(xf)
        xe = self.redu_e(xe)
        b, c, h, w = xf.shape
        xf = xf.view(1, b * c, h, w)
        xe = xe.view(1, b * c, h, w)
        ker_f = ker_f.view(b * c, 1, self.filter_size, self.filter_size)
        ker_e = ker_e.view(b * c, 1, self.filter_size, self.filter_size)

        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        xf = F.pad(input=xf, pad=p2d, mode='constant', value=0)
        xe = F.pad(input=xe, pad=p2d, mode='constant', value=0)
        # [1, b * c, h, w]
        guide_f = F.conv2d(input=xf, weight=ker_e, groups=b * c)
        guide_e = F.conv2d(input=xe, weight=ker_f, groups=b * c)
        guide_f = guide_f.view(b, c, h, w)
        guide_e = guide_e.view(b, c, h, w)
        guide_f = self.bottle_f(guide_f)
        guide_e = self.bottle_e(guide_e)
        return guide_f, guide_e


class Multi_Context(nn.Module):
    def __init__(self, inchannels):
        super(Multi_Context, self).__init__()
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels * 3, out_channels=inchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannels))

    def forward(self, x):
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        x = self.conv2(x)
        return x

class Adaptive_Weight(nn.Module):
    def __init__(self, inchannels):
        super(Adaptive_Weight, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.inchannels = inchannels
        self.fc1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inchannels//4, 1, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg(x)
        weight = self.relu1(self.fc1(x_avg))
        weight = self.relu2(self.fc2(weight))
        weight = self.sigmoid(weight)
        out = x * weight
        return out

class FUSION(nn.Module):
    def __init__(self, inchannels):
        super(FUSION, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inchannels))
        self.sig = nn.Sigmoid()
        self.mc1 = Multi_Context(inchannels)
        self.mc2 = Multi_Context(inchannels)
        self.ada_w1 = Adaptive_Weight(inchannels)
        self.ada_w2 = Adaptive_Weight(inchannels)

    def forward(self, assistant, present):
        mc1 = self.mc1(assistant)
        pr1 = present * self.sig(mc1)
        pr2 = self.conv1(present)
        pr2 = present * self.sig(pr2)
        out1 = pr1 + pr2 + present

        mc2 = self.mc2(present)
        as1 = assistant * self.sig(mc2)
        as2 = self.conv2(assistant)
        as2 = assistant * self.sig(as2)
        out2 = as1 + as2 + assistant

        out1 = self.ada_w1(out1)
        out2 = self.ada_w2(out2)
        out = out1 + out2

        return out

class ALIGN(nn.Module):

    def __init__(self, input_dim):
        """
        input:
        output:
        """

        super(ALIGN, self).__init__()

        self.off2d_1 = nn.Conv2d(input_dim, input_dim, 3, padding=1, bias=True)
        self.dconv_1 = DCNv2Pack(input_dim, input_dim, 3, padding=1, deformable_groups=8)

    def forward(self, cat_fea, f_fea):
        offset1 = self.off2d_1(cat_fea)
        fea = (self.dconv_1(f_fea, offset1))
        aligned_fea = fea
        return aligned_fea

class ALIGN_FUSION(nn.Module):
    def __init__(self):
        super(ALIGN_FUSION, self).__init__()
        ## for align
        self.guide_high = EgF(inplanes=256, planes=256//16)
        self.guide_low = EgF(inplanes=128, planes=128 // 16)
        self.align_high = ALIGN(input_dim=256)
        self.align_low = ALIGN(input_dim=128)

        ## for dynamic fusion
        self.fusion_high = Fusion_dynamic(n_feat=256)
        self.fusion_low = Fusion_dynamic(n_feat=128)
        self.b1 = nn.Conv2d(256*2, 256, 1)
        self.b2 = nn.Conv2d(128*2, 128, 1)

    def forward(self, high_frame, low_frame, high_event, low_event):
        """  """
        high_frame, fea_e1 = self.guide_high(high_frame, high_event)
        fea_e1 = fea_e1 + high_event
        high_fea = adain(fea_e1, high_frame)
        cat_fea = self.b1(torch.cat([high_fea, fea_e1], dim=1))
        align_high = self.align_high(cat_fea, high_frame)
        fuse_high = self.fusion_high(align_high, fea_e1)

        low_frame, fea_e2  = self.guide_low(low_frame, low_event)
        fea_e2 = fea_e2 + low_event
        low_fea = adain(fea_e2, low_frame)
        cat_fea = self.b2(torch.cat([low_fea, fea_e2], dim=1))
        align_low = self.align_low(cat_fea, low_frame)
        fuse_low  = self.fusion_low(align_low, fea_e2)

        return  fuse_low + low_event, fuse_high + high_event

if __name__ == '__main__':
    channels = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = ALIGN_FUSION()
    model = model.cuda()
    high_frame = torch.FloatTensor(2, 256, 18, 18).cuda()
    low_frame = torch.FloatTensor(2, 128, 36, 36).cuda()
    high_event = torch.FloatTensor(2, 256, 18, 18).cuda()
    low_event = torch.FloatTensor(2, 128, 36, 36).cuda()
    layer_output_list, last_state_list = model(high_frame, low_frame, high_event, low_event)
    print(layer_output_list.shape, last_state_list.shape)
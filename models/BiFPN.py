# Reference : https://github.com/Michael-Jing/EfficientDet-pytorch/blob/master/efficientdet_pytorch/BiFPN.py

import torch
from torch import nn
from torch.nn import functional as F


def resize(input_tensor, size):
    return F.upsample(input_tensor, size=size, mode='bilinear')

class Con2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, input):
        x = self.conv2d(input)
        x = self.bn(x)
        return F.relu(x)

class BiFPNBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels, gpu_id=0):
        # TODO:
        # determine the number of in_channels 
        super().__init__()
        n_gpus = torch.cuda.device_count()
        gpu_id = (n_gpus-1) if gpu_id >= n_gpus else gpu_id
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p4_mid_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p4_mid_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)

        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)
        self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True).cuda(device=gpu_id)

        self.p3_in_conv = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)
        self.p4_in_conv = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)
        self.p5_in_conv = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)

        self.p4_mid_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=out_channels, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)

        self.p3_out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=out_channels, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)
        self.p4_out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=out_channels, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)
        self.p5_out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=out_channels, dilation=1, bias=True, padding_mode='zeros').cuda(device=gpu_id)


    def forward(self, input):
        epsilon = 0.0001

        if (type(input) != type(list())):
            p3, p4, p5 = list(input.values())
        else:
            p3, p4, p5 = input

        p3 = self.p3_in_conv(p3)
        p4 = self.p4_in_conv(p4)
        p5 = self.p5_in_conv(p5)

        size_of_p3 = p3.shape[2:]
        size_of_p4 = p4.shape[2:]
        size_of_p5 = p5.shape[2:]

        p4_mid = self.p4_mid_conv((self.p4_mid_w1 * p4 + self.p4_mid_w2 * resize(p5, size_of_p4)) /
                                   (self.p4_mid_w1 + self.p4_mid_w2 + epsilon))

        p3_out = self.p3_out_conv((self.p3_out_w1 * p3 + self.p3_out_w2 * resize(p4_mid, size_of_p3)) /
                                    (self.p3_out_w1 + self.p3_out_w2 + epsilon))
        p4_out = self.p4_out_conv((self.p4_out_w1 * p4 + self.p4_out_w2 * p4_mid + self.p4_out_w3 * resize(p3_out, size_of_p4))
                                / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
        p5_out = self.p5_out_conv((self.p5_out_w1 * p5 + self.p5_out_w3 * resize(p4_out, size_of_p5))
                                / (self.p5_out_w1 + self.p5_out_w3 + epsilon))
                                
        return [p3_out, p4_out, p5_out]

class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, n_stack=3):
        super().__init__()

        self.FPNs = nn.ModuleList()
        self.n_stack = n_stack

        for i in range(self.n_stack):
            in_channels = in_channels_list if i==0 else [out_channels]*len(in_channels_list)
            self.FPNs.append(BiFPNBlock(in_channels, out_channels, gpu_id=i))

    def forward(self, x):
        for i in range(self.n_stack):
            x = self.FPNs[i](x)
        return x

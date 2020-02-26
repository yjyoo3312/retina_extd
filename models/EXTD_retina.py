import torch
import torch.nn as nn
import torch.nn.functional as F

from models.net import FPN as FPN
from models.net import SSH as SSH

def Swish(x):

    return x*torch.sigmoid(x)

class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


def conv_bn(inp, oup, stride, k_size=3):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup, momentum=0.01),
        nn.PReLU()
    )

class EXTD_retina(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(EXTD_retina, self).__init__()
        self.phase = phase

        feat_ch = cfg['in_channel']

        # body will contain the list of the channel
        self.base = []
        # backbone network
        self.base.append(conv_bn(inp=3, oup=feat_ch, stride=2, k_size=3))

        # for conv - 1
        self.base.append(nn.Conv2d(feat_ch, feat_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
        # bn
        self.base.append(nn.PReLU(num_parameters=1))

        self.base.append(nn.Conv2d(feat_ch, feat_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        # bn

        # for shortcut
        self.base.append(nn.Conv2d(feat_ch, feat_ch, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False))
        # bn

        self.base.append(nn.PReLU(num_parameters=1))

        # for conv - 2
        self.base.append(nn.Conv2d(feat_ch, feat_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        # bn
        self.base.append(nn.PReLU(num_parameters=1))

        self.base.append(nn.Conv2d(feat_ch, feat_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        # bn

        self.num_bn = 5

        self.base = nn.ModuleList(self.base)

        self.bns = []

        for r in range(6):  # num of
            for i in range(self.num_bn):
                self.bns.append(
                    nn.BatchNorm2d(feat_ch, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True))

        self.bns = nn.ModuleList(self.bns)


        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2,
            in_channels_stage2,
        ]
        out_channels = in_channels_list
        self.fpn = FPN(in_channels_list, out_channels) # we use five fpn
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=5, inchannels=cfg['out_channel'])

    def base_block(self, x, bn_list):

        # short cut block
        sc = bn_list[0](self.base[4](x))  # conv s=2 p shortcut

        # conv block
        out = self.base[1](x)  # conv(x) s = 2
        out = bn_list[1](out)  # bn(x)
        out = self.base[2](out)  # PReLU(x)

        out = self.base[3](out)  # conv(x) s = 1
        out = bn_list[2](out)  # bn(x)

        x2 = self.base[5](out + sc)  # PReLU()

        # conv block
        out = self.base[6](x2)  # conv(x) s = 1
        out = bn_list[3](out)  # bn(x)
        out = self.base[7](out)  # PReLU(x)

        out = self.base[8](out)  # conv(x) s = 1
        out = bn_list[4](out)  # bn(x)

        return x2 + out  # skip connection

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):

        out = []
        # pass the first filter
        x = self.base[0](inputs)
        # iteration 1
        x = self.base_block(x, self.bns[:self.num_bn])
        out.append(x)
        # 2 iteration
        x = self.base_block(Swish(x), self.bns[self.num_bn:2 * self.num_bn])
        out.append(x)
        # 3 iteration
        x = self.base_block(Swish(x), self.bns[2 * self.num_bn:3 * self.num_bn])
        out.append(x)
        # 4 iteration
        x = self.base_block(Swish(x), self.bns[3 * self.num_bn:4 * self.num_bn])
        out.append(x)
        # 5 iteration
        x = self.base_block(Swish(x), self.bns[4 * self.num_bn:5 * self.num_bn])
        out.append(x)


        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        feature4 = self.ssh3(fpn[3])
        feature5 = self.ssh3(fpn[4])
        features = [feature1, feature2, feature3, feature4, feature5]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
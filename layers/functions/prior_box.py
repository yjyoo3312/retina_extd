import torch
from itertools import product as product
import numpy as np
from math import ceil, floor


class PriorBox(object):
    def __init__(self, cfg, aspect_ratio=1.25, image_size=None, phase='train', mode='ceil'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.scales = cfg['scales']
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size

        if mode == 'ceil':
            self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        elif mode == 'floor':
            self.feature_maps = [[floor(self.image_size[0]/step), floor(self.image_size[1]/step)] for step in self.steps]

        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            anchor_scales = self.scales[k]
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    for scale in anchor_scales:
                        s_kx = min_size / self.image_size[1] * scale
                        s_ky = min_size / self.image_size[0] * scale * self.aspect_ratio
                        cx = (j+0.5) * self.steps[k] / self.image_size[1]
                        cy = (i+0.5) * self.steps[k] / self.image_size[0]
                        anchors.append([cx, cy, s_kx, s_ky])
         
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

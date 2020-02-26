import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from glob import glob
from math import floor, ceil

from scipy.io import loadmat

class AFLW(data.Dataset):
    def __init__(self, root, target_size, subset='test', rgb_mean=(104, 117, 123), landmark_idxs=[9, 12, 15, 17, 19]):
        super(AFLW, self).__init__()
        self.subset = subset
        self.landmark_idxs = landmark_idxs
        self.rgb_mean = rgb_mean
        self.target_size = target_size

        self.img_paths = glob(os.path.join(root, subset, '*', '*_*.png'))

        print('>>> Done in Constructing AFLW %s dataset, #samples: %d' % \
              (self.subset, self.__len__()))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt = np.load(img_path.replace('.png', '_gt.npy'))

        img = cv2.imread(img_path)
        img, gt = self.resized_and_pad_image(img, gt, self.target_size)

        img = img - self.rgb_mean

        target_width, target_height = self.target_size

        annotation = np.zeros((1, 15))

        # bbox
        annotation[0, 0] = int(gt[0]) / target_width  # x1
        annotation[0, 1] = int(gt[1]) / target_height  # y1
        annotation[0, 2] = int(gt[2]) / target_width  # x2
        annotation[0, 3] = int(gt[3]) / target_height  # y2

        # landmarks
        for i, idxs in enumerate(self.landmark_idxs):
            annotation[0, 2*i+4]  = gt[2*idxs] / target_width  # lx
            annotation[0, 2*i+5]  = gt[2*idxs+1] / target_height    # lx

        if (annotation[0, 4]<0):
            annotation[0, 14] = -1
        else:
            annotation[0, 14] = 1

        img = np.moveaxis(img, -1, 0)
        target = np.array(annotation)

        return torch.from_numpy(img), target, img_path
    
    def resized_and_pad_image(self, img, gt, target_size=(256, 256)):
        w_target, h_target = target_size

        h, w, _ = img.shape
        long_axis = max(h, w)

        img_resized = cv2.resize(img, (int(w/long_axis*w_target), int(h/long_axis*h_target)))  # Maintain width/height ratio
        h_resized, w_resized, _ = img_resized.shape

        w_diff_half = (w_target - w_resized) / 2
        h_diff_half = (h_target - h_resized) / 2

        w_pad = (floor(w_diff_half), ceil(w_diff_half))
        h_pad = (floor(h_diff_half), ceil(h_diff_half))
        img_padded = np.pad(img_resized, (h_pad, w_pad, (0,0)), mode='constant')
        gt_moved = gt.copy()
        gt_moved[range(0, len(gt_moved), 2)] = (gt_moved[range(0, len(gt_moved), 2)] / long_axis * w_target + w_pad[0])
        gt_moved[range(1, len(gt_moved), 2)] = (gt_moved[range(1, len(gt_moved), 2)] / long_axis * h_target + h_pad[0])
        return img_padded, gt_moved
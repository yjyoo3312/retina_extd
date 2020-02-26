from __future__ import print_function

import os
import time
import argparse
import datetime
import numpy as np
from glob import glob

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.utils.data as data

from data import *
from models import check_latency, load_network

import warnings
warnings.filterwarnings('ignore')

# NSML
try:
  import nsml
  from nsml import DATASET_PATH, NSML_NFS_OUTPUT
  USE_NSML = True
  print('\nThis script will be ran on the NSML')
except ImportError as e:
  USE_NSML = False
  print('\nThis script will be ran on the local machine')

# Args
parser = argparse.ArgumentParser(description='Retinaface Latency')

parser.add_argument('--image_width', default=320, type=int)
parser.add_argument('--image_height', default=320, type=int)

parser.add_argument('--network', default='resnet50-3fpn', help='resnet50-3fpn, resnet50-5fpn, mobile0.25, efficienttiny')
parser.add_argument('--FPN', default='FPN', help='FPN, BiFPN')
parser.add_argument('--repeat', default=500, type=int)
parser.add_argument('--replace_denormals', action='store_true')
parser.add_argument('--bn_fold', action='store_true')
parser.add_argument('--pretrained_backbone', default='', help='pretrained pth path for backbone network')

args = parser.parse_args()


# Load Configuration
net, cfg = load_network(args)
check_latency(net, 3, args.image_height, args.image_width, bn_fold=args.bn_fold, repeat=args.repeat, replace_denormals=args.replace_denormals)

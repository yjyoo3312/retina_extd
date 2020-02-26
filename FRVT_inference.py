'''
Hard Case Images
1. S093-01-t10_01.png
2. S300-01-t10_01.png
3. rotated90.png
4. rotated180.png
'''
import os
import cv2
import sys
sys.path.append('../RetinaFace')

from glob import glob
from tqdm import tqdm

import torch
import numpy as np
from models import load_network
from shutil import rmtree

from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
import argparse

parser = argparse.ArgumentParser(description='Retinaface FRVT Inference')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50 or mobilelight')
parser.add_argument('--FPN', default='FPN', help='FPN, BiFPN')
parser.add_argument('--FPN_stack', default=3, type=int, help='number of FPN layers (only valid for BiFPN)')
parser.add_argument('--img_dim', default=320, type=int)
parser.add_argument('--save_size', default=320, type=int)
parser.add_argument('--confidence_threshold', default=0.5, type=float)
parser.add_argument('--nms_threshold', default=0.7, type=float)
parser.add_argument('--save_dir', default='/home/dongmin/FRVT', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', default='', help='trained pth path for inference')
parser.add_argument('--pretrained_backbone', default='', help='pretrained pth path for backbone network')
args = parser.parse_args()

args.resume = os.path.join('/home/dongmin/%s' % ('{}_weights.pth'.format(args.network) if args.resume == '' else args.resume))
img_list = glob('../FRVT/images/*.ppm')
save_dir = os.path.join(args.save_dir, '%s_size_%d_result' % (args.network, args.img_dim))

if os.path.exists(save_dir):
    rmtree(save_dir)
    os.makedirs(save_dir)

img_dim = args.img_dim
rgb_mean = (104, 117, 123) # bgr order

net, cfg = load_network(args)

train_state_dict = torch.load(args.resume, map_location=torch.device(args.device))

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in train_state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
train_state_dict = new_state_dict

net.load_state_dict(train_state_dict)
net.to(torch.device(args.device))

top_k = 5000
keep_top_k = 750
vis_thres = 0.0

for img_path in tqdm(img_list):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    long_len = max([height, width])
    ratio = long_len/img_dim
    
    img_resized = cv2.resize(img, (int(width/ratio), int(height/ratio)))
    img_resized = img_resized - rgb_mean
    img_input = torch.Tensor(np.moveaxis(img_resized, -1, 0)[None, ...])
    
    if args.device == 'cuda':
        img_input = img_input.to(torch.device(args.device))
        with torch.no_grad():
            pred = net(img_input)
    
    img_resized = img_resized + rgb_mean
    img_resized = img_resized.copy().astype(np.uint8)
    loc, conf, landms = [pred[0][0].cpu(), pred[1][0].cpu(), pred[2][0].cpu()]
    priorbox = PriorBox(cfg, image_size=(int(height/ratio), int(width/ratio)), mode='floor')
    priors = priorbox.forward()
    prior_data = priors.data
    boxes = decode(loc.data, prior_data, cfg['variance'])
    boxes = boxes.cpu().numpy()
    conf = conf.softmax(dim=1)[:, 1]
    scores = conf.data.cpu().numpy()
    landms = decode_landm(landms.data, prior_data, cfg['variance'])
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    h_resized, w_resized, _ = img_resized.shape
    
    n_pred = len(dets)

    # show image
    for b in dets:
        if b[4] < vis_thres:
            continue

        b[[0, 2, 5, 7, 9, 11, 13]] *= width
        b[[1, 3, 6, 8, 10, 12, 14]] *= height
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)

    save_folder = os.path.join(save_dir, '%d_detected_vis' % n_pred)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_ratio = long_len/args.save_size
    img = cv2.resize(img, (int(width/save_ratio), int(height/save_ratio)))
    cv2.imwrite(os.path.join(save_folder, img_path.split(os.sep)[-1].replace('ppm', 'png')), img)
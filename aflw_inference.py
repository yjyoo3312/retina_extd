from __future__ import print_function
import os
import numpy as np
from glob import glob
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import AFLW, detection_collate, preproc, cfg_mnet, cfg_re50_3fpn, cfg_re50_5fpn, cfg_mnetlgt, cfg_mnetlgtv2
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from models import check_latency
from utils import AverageMeter
from utils.nme import NME
from utils.vis import visualize_batch_result


import warnings
warnings.filterwarnings('ignore')

try:
  import nsml
  from nsml import DATASET_PATH, NSML_NFS_OUTPUT
  USE_NSML = True
  print('\nThis script will be ran on the NSML')
except ImportError as e:
  USE_NSML = False
  print('\nThis script will be ran on the local machine')

print("Device : %s \n" % torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--aflw_dataset', default='taey16/storage/AFLW_cropped_ratio_0.15', help='Validation dataset (AFLW) directory')
parser.add_argument('--aflw_size', default=256, type=int, help='AFLW face ROI resize shape')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50 or mobilelight')
parser.add_argument('--fpn_num', default=5, type=int, help='number of fpn layer (3 : P2 ~ P4 / 5 : P2 ~ P6)')
parser.add_argument('--aspect_ratio', default=1.25, type=float, help='Anchor box aspect ratio')
parser.add_argument('--num_workers', default=10, type=int, help='Number of workers used in dataloading')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_folder', default='dongmin/retinaface_ckpt/', help='Location to save checkpoint models')
parser.add_argument('--check_latency', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--resnet_pretrained_path', default='dongmin/retinaface_ckpt/resnet50-19c8e357.pth', help='Imagenet pretrained resnet path')

args = parser.parse_args()

args.aflw_dataset = os.path.join('/home', args.aflw_dataset) if not USE_NSML else os.path.join(DATASET_PATH[1], args.aflw_dataset)
args.save_folder = os.path.join('/home', args.save_folder) if not USE_NSML else os.path.join(NSML_NFS_OUTPUT, args.save_folder)


if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50" and args.fpn_num == 5:
    cfg = cfg_re50_5fpn
elif args.network == "resnet50" and args.fpn_num == 3:
    cfg = cfg_re50_3fpn
elif args.network == "mobilelight":
    cfg = cfg_mnetlgt
elif args.network == "mobilelightv2":
    cfg = cfg_mnetlgtv2

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
batch_size = 64
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']
aflw_target_size = (args.aflw_size, args.aflw_size)





# Model
cfg['pretrain'] = False
net = RetinaFace(args, cfg=cfg, fpn_num=args.fpn_num)
if args.check_latency:
    check_latency(net, 3, img_dim, img_dim)


if args.resume_net is not None:
    args.resume_net = os.path.join('/home', args.resume_net) if not USE_NSML else os.path.join(NSML_NFS_OUTPUT, args.resume_net)
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    print("Successfully loaded trained weights from %s" % args.resume_net)

if torch.cuda.device_count() > 1 and gpu_train:
    print("Multi-gpu Training (%d GPUs)" % torch.cuda.device_count())
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True



# Main
def main():
    # Priorbox Setting
    priorbox = PriorBox(cfg, aspect_ratio=args.aspect_ratio, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    print('Loading Dataset...')
    aflw_dataset = AFLW(args.aflw_dataset, target_size=aflw_target_size, subset='test', rgb_mean=rgb_mean)
    # Priorbox Setting
    aflw_priorbox = PriorBox(cfg, aspect_ratio=args.aspect_ratio, image_size=(args.aflw_size, args.aflw_size))
    with torch.no_grad():
        aflw_priors = aflw_priorbox.forward()
        aflw_priors = aflw_priors.cuda()

    # ================================= Eval AFLW =================================
    print("\nStart AFLW Validation...")
    # loss_l_meter, loss_c_meter, loss_landm_meter, loss_total_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    aflw_batch_iterator = iter(data.DataLoader(aflw_dataset, batch_size, shuffle=False, num_workers=args.num_workers))
    nme_result = np.zeros(5)
    aflw_iter_cnt = 0
    epoch = args.resume_epoch

    for i in range(len(aflw_batch_iterator)):
        # load data
        images, targets, img_paths = next(aflw_batch_iterator)
        images = images.cuda().float()
        height, width = images.size()[2:]

        targets = targets.cuda().float()

        # forward
        with torch.no_grad():
            out = net(images)

        # calculate loss
        # loss_l, loss_c, loss_landm = criterion(out, aflw_priors, targets)
        # loss = cfg['loc_weight'] * loss_l + loss_c + cfg['landm_weight'] * loss_landm

        # calculate nme
        nme_result += NME(out, targets, aflw_priors, cfg)

        # update result
        # loss_l_meter.update(loss_l.item(), n=len(images))
        # loss_c_meter.update(loss_c.item(), n=len(images))
        # loss_landm_meter.update(loss_landm.item(), n=len(images))
        # loss_total_meter.update(loss.item(), n=len(images))

        if (i % 9) == 0:
            print(">>> [%3d/%3d] Validated..." % (i+1, len(aflw_batch_iterator)))
        if args.visualize:
            dir_name = '%04d_epoch_vis' % (epoch+1)
            vis_save_dir = os.path.join(args.save_folder, dir_name)
            aflw_iter_cnt = visualize_batch_result(images, out, img_paths, cfg, aflw_iter_cnt, save_dir=vis_save_dir)
    
    # aflw_loss_l, aflw_loss_c, aflw_loss_landm, aflw_loss_total = loss_l_meter.avg, loss_c_meter.avg, loss_landm_meter.avg, loss_total_meter.avg
    nme_result = nme_result / len(aflw_batch_iterator)
    aflw_nme = np.mean(nme_result)

    print(">>> Epoch [%3d] Result = Left eye %.4f - Right eye %.4f - Nose %.4f - Left mouth %.4f - Right mouth %.4f - Total %.4f"
                % (epoch, *nme_result, aflw_nme))
            
    print()

if __name__ == '__main__':
    main()

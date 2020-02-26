from __future__ import print_function

import os
import time
import argparse
import datetime
import numpy as np
from glob import glob

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.utils.data as data
import torch.optim as optim

from data import *
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from utils.nme import NME
from utils import AverageMeter, lr_update, load_optimizer
from utils.vis import visualize_batch_result
from models import EXTD_retina

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
parser = argparse.ArgumentParser(description='Retinaface Training')

parser.add_argument('--training_dataset', default='taey16/storage/widerface/WIDER_train/label.txt', help='Training dataset (Wider Face) directory')
parser.add_argument('--validation_dataset', default='taey16/storage/widerface/WIDER_val/label.txt', help='Validation dataset (Wider Face) directory')
parser.add_argument('--aflw_dataset', default='taey16/storage/AFLW_cropped_ratio_0.15', help='Validation dataset (AFLW) directory')
parser.add_argument('--image_size', default=640, type=int)
parser.add_argument('--aflw_size', default=256, type=int, help='AFLW face ROI resize shape')
parser.add_argument('--num_workers', default=10, type=int, help='Number of workers used in dataloading')

parser.add_argument('--network', default='extd', help='resnet50-3fpn, resnet50-5fpn, mobile0.25, efficientb0-3fpn, efficientb0-3fpnv2, efficientb0-5fpn, efficientb1-3fpn, extd, extd_non')
parser.add_argument('--FPN', default='FPN', help='FPN, BiFPN')
parser.add_argument('--FPN_stack', default=3, type=int, help='number of FPN layers (only valid for BiFPN)')
parser.add_argument('--aspect_ratio', default=1.25, type=float, help='Anchor box aspect ratio')
parser.add_argument('--pretrained_backbone', default='', help='pretrained pth path for backbone network')

parser.add_argument('--max_epoch', default=300, type=int)
parser.add_argument('--batch_size', default=16, type=int)

parser.add_argument('--optim', default='SGD', help='SGD, Adam, AdamW, rmsprop')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--lr_warmup_epoch', default=5, type=int, help='Rising to 10x lr after which epoch')
parser.add_argument('--lr_decay_epoch', default='10,55,130,200', type=str, help='Multiply 0.1 to lr at which epoch')
parser.add_argument('--loc_weight', default=2.0, type=float, help='lambda weight for localization loss term')
parser.add_argument('--landm_weight', default=2.0, type=float, help='lambda weight for landmark loss term')
parser.add_argument('--neg_pos_ratio', default=3.0, type=float, help='Negative anchor : positive anchor')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')

parser.add_argument('--resume_net', default='', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')

parser.add_argument('--save_folder', default='dongmin/retinaface_ckpt/', help='Location to save checkpoint models')
parser.add_argument('--visualize', action='store_true')

args = parser.parse_args()

args.training_dataset = os.path.join('/home', args.training_dataset) if not USE_NSML else os.path.join(DATASET_PATH[1], args.training_dataset)
args.validation_dataset = os.path.join('/home', args.validation_dataset) if not USE_NSML else os.path.join(DATASET_PATH[1], args.validation_dataset)
args.aflw_dataset = os.path.join('/home', args.aflw_dataset) if not USE_NSML else os.path.join(DATASET_PATH[1], args.aflw_dataset)
args.pretrained_backbone = os.path.join('/home', args.pretrained_backbone) if not USE_NSML else os.path.join(NSML_NFS_OUTPUT, args.pretrained_backbone)
args.resume_net = os.path.join('/home', args.resume_net) if not USE_NSML else os.path.join(NSML_NFS_OUTPUT, args.resume_net)
args.save_folder = os.path.join('/home', args.save_folder) if not USE_NSML else os.path.join(NSML_NFS_OUTPUT, args.save_folder)

# Make Save Dir
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


rgb_mean = (104, 117, 123) # bgr order
num_classes = 2


# Model
cfg = cfg_extd
net = EXTD_retina(cfg=cfg)
print("Printing net...")
print(net)

#lr setting
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

# Compile
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

# Main
def main():
    # Priorbox Setting
    priorbox = PriorBox(cfg, aspect_ratio=args.aspect_ratio, image_size=(args.image_size, args.image_size))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    
    aflw_priorbox = PriorBox(cfg, aspect_ratio=args.aspect_ratio, image_size=(args.aflw_size, args.aflw_size))
    with torch.no_grad():
        aflw_priors = aflw_priorbox.forward()
        aflw_priors = aflw_priors.cuda()

    # Log
    trainLogger = open('%s/train.log' % args.save_folder, 'w')
    valLogger   = open('%s/val.log' % args.save_folder, 'w')
    aflwLogger  = open('%s/aflw.log' % args.save_folder, 'w')
    
    # Dataset
    best_loss_epoch = -1
    best_nme_epoch = -1
    best_loss = 100000.
    best_nme = 100000.
    global_iter = 0
    print('Loading Dataset...')

    train_dataset = WiderFaceDetection(args.training_dataset, preproc(args.image_size, rgb_mean, subset='train'))
    valid_dataset = WiderFaceDetection(args.validation_dataset, preproc(args.image_size, rgb_mean, subset='test'), 'test')

    aflw_dataset = AFLW(args.aflw_dataset, target_size=(args.aflw_size, args.aflw_size), subset='test', rgb_mean=rgb_mean)
    

    print("\nStart Training...")
    for epoch in range(args.max_epoch):
        epoch += args.resume_epoch
        lr_update(epoch, args, optimizer)

        # ================================= Train =================================
        net.train()

        loss_l_meter, loss_c_meter, loss_landm_meter, loss_total_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        train_batch_iterator = iter(data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
        
        for i in range(len(train_batch_iterator)):
            load_t0 = time.time()

            # load data
            images, targets = next(train_batch_iterator)

            # ====================================================== Data Loader Check ======================================================
            import cv2

            def annot(img, gts, rot_img=False):
                img_annot = img.copy()
                
                for gt in gts:
                    x_min, y_min, x_max, y_max = gt[:4].astype(int)
                    landmarks = np.array(gt[4:]).astype(int)

                    for i in range(0, len(landmarks), 2):
                        cv2.circle(img_annot, (landmarks[i], landmarks[i+1]), 1, (0, 0, 255), 4)
                    
                    if rot_img:
                        cv2.circle(img_annot, (x_min, y_min), 1, (255, 0, 0), 4)
                        cv2.circle(img_annot, (x_max, y_max), 1, (255, 0, 0), 4)
                        
                    else:
                        cv2.rectangle(img_annot, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=3)

                return img_annot
            
            if False:
                images_array = np.moveaxis(images.numpy(), 1, -1)

                if not os.path.exists('widerface_dataloader'):
                    os.makedirs('widerface_dataloader')

                for j in range(len(images)):
                    img = images_array[j] + rgb_mean
                    img = img.astype(np.uint8).copy()
                    target = targets[j][:, :14].numpy() * 640
                    image_annoted = annot(img, target, rot_img=True)
                    
                    cv2.imwrite('widerface_dataloader/epoch_%03d_iter_%03d_%2d_image.png' % (epoch+1, i, j), image_annoted)
            # ==================================================================================================================================



            images = images.cuda()
            targets = [anno.cuda() for anno in targets]



            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = args.loc_weight * loss_l + loss_c + args.landm_weight * loss_landm

            loss.backward()
            optimizer.step()

            # update result
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (len(train_batch_iterator) - i-1))

            loss_l_meter.update(loss_l.item(), n=len(images))
            loss_c_meter.update(loss_c.item(), n=len(images))
            loss_landm_meter.update(loss_landm.item(), n=len(images))
            loss_total_meter.update(loss.item(), n=len(images))

            if (i % 29) == 0:
                print('Epoch:{%3d}/{%3d} || Epochiter: {%4d}/{%4d} || Loc: {%7.4f} Cla: {%7.4f} Landm: {%7.4f} Total: {%7.4f} || LR: {%f} || Batchtime: {%7.4f} s || ETA: {%s}'
                    % (epoch+1, args.max_epoch, i+1, len(train_batch_iterator),
                    loss_l_meter.avg, loss_c_meter.avg, loss_landm_meter.avg, loss_total_meter.avg, args.lr, batch_time, str(datetime.timedelta(seconds=eta))))
                
            trainLogger.write('iter\t%d\tloc\t%f\tcls\t%f\tpts\t%f\ttotal\t%f\n' % (global_iter, loss_l_meter.avg, loss_c_meter.avg, loss_landm_meter.avg, loss_total_meter.avg))
            global_iter += 1

        train_loss_l, train_loss_c, train_loss_landm, train_loss_total = loss_l_meter.avg, loss_c_meter.avg, loss_landm_meter.avg, loss_total_meter.avg
        



        # ================================= Eval WiderFace =================================
        net.eval()
        print("\nStart WiderFace Validation...")

        loss_l_meter, loss_c_meter, loss_total_meter = AverageMeter(), AverageMeter(), AverageMeter()
        valid_batch_iterator = iter(data.DataLoader(valid_dataset, args.batch_size*2, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate))

        for i in range(len(valid_batch_iterator)):
            # load data
            images, targets = next(valid_batch_iterator)
            images = images.cuda()
            targets = [anno.cuda().float() for anno in targets]

            # forward
            with torch.no_grad():
                out = net(images)
            
            # calculate loss
            loss_l, loss_c, _ = criterion(out, priors, targets)
            loss = args.loc_weight * loss_l + loss_c

            # update result
            loss_l_meter.update(loss_l.item(), n=len(images))
            loss_c_meter.update(loss_c.item(), n=len(images))
            loss_total_meter.update(loss.item(), n=len(images))

            if (i % 9) == 0:
                print(">>> [%3d/%3d] Validated..." % (i+1, len(valid_batch_iterator)))

        valid_loss_l, valid_loss_c, valid_loss_total = loss_l_meter.avg, loss_c_meter.avg, loss_total_meter.avg



        # ================================= Eval AFLW =================================
        print("\nStart AFLW Validation...")
        aflw_batch_iterator = iter(data.DataLoader(aflw_dataset, args.batch_size*2, shuffle=False, num_workers=args.num_workers))
        nme_result = np.zeros(5)
        aflw_iter_cnt = 0

        for i in range(len(aflw_batch_iterator)):
            # load data
            images, targets, img_paths = next(aflw_batch_iterator)
            images = images.cuda().float()
            height, width = images.size()[2:]

            targets = targets.cuda().float()

            # forward
            with torch.no_grad():
                out = net(images)

            # calculate nme
            nme_result += NME(out, targets, aflw_priors, cfg)

            if (i % 9) == 0:
                print(">>> [%3d/%3d] Validated..." % (i+1, len(aflw_batch_iterator)))

            if args.visualize:
                dir_name = '%04d_epoch_vis' % (epoch+1)
                vis_save_dir = os.path.join(args.save_folder, dir_name)
                aflw_iter_cnt = visualize_batch_result(images, out, img_paths, cfg, aflw_iter_cnt, save_dir=vis_save_dir)
        
        nme_result = nme_result / len(aflw_batch_iterator)
        aflw_nme = np.mean(nme_result)

        valLogger.write('epoch\t%d\tloc\t%f\tcls\t%f\ttotal\t%f\n' % (epoch, valid_loss_l, valid_loss_c, valid_loss_total))
        aflwLogger.write('epoch\t%d\tnme\t%f\n' % (epoch, aflw_nme))

        trainLogger.flush()
        valLogger.flush()
        aflwLogger.flush()

        # Best Performance Check
        if valid_loss_total < best_loss:
            best_loss_epoch = epoch+1
            best_loss = valid_loss_total
            [os.remove(path) for path in glob(os.path.join(args.save_folder, '%s_best_loss*.pth' % cfg['name']))] # Remove previous weights

            print("Best Loss Updated...")
            torch.save(net.state_dict(),
                    os.path.join(args.save_folder, cfg['name'] + '_best_loss_epoch_%d_loc_%.4f_cla_%.4f_aflw_nme_%.4f_total_%.4f.pth' % (epoch+1, valid_loss_l, valid_loss_c, np.mean(nme_result), valid_loss_total)))
        
        if aflw_nme < best_nme:
            best_nme_epoch = epoch+1
            best_nme = aflw_nme
            [os.remove(path) for path in glob(os.path.join(args.save_folder, '%s_best_nme*.pth' % cfg['name']))] # Remove previous weights

            print("Best NME updated...")
            torch.save(net.state_dict(),
                    os.path.join(args.save_folder, cfg['name'] + '_best_nme_epoch_%d_loc_%.4f_cla_%.4f_aflw_nme_%.4f_total_%.4f.pth' % (epoch+1, valid_loss_l, valid_loss_c, np.mean(nme_result), valid_loss_total)))
        

        print(">>> Epoch [%3d/%3d] Result = Train Loc : %7.4f - Train Cla : %7.4f - Train Landm : %7.4f - Train Total : %7.4f\n"
              "                             Valid Loc : %7.4f - Valid Cla : %7.4f - Valid Total : %7.4f\n"
              ">>> NME Result = Left eye %.4f - Right eye %.4f - Nose %.4f - Left mouth %.4f - Right mouth %.4f - Total %.4f\n"
              ">>> Best Loss Epoch = %d - Best Loss %7.4f\n"
              ">>> Best NME  Epoch = %d - Best NME %.4f"
                    % (epoch+1, args.max_epoch, train_loss_l, train_loss_c, train_loss_landm, train_loss_total,
                       valid_loss_l, valid_loss_c, valid_loss_total,
                        *nme_result, aflw_nme,
                        best_loss_epoch, best_loss,
                        best_nme_epoch, best_nme))

        print()
        if (epoch+1) == args.max_epoch:
            break
    
    trainLogger.close()
    valLogger.close()
    aflwLogger.close()



if __name__ == '__main__':
    main()
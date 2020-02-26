from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import *
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models import load_network
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from tqdm import tqdm
# NSML
try:
  import nsml
  from nsml import DATASET_PATH, NSML_NFS_OUTPUT
  USE_NSML = True
  print('\nThis script will be ran on the NSML')
except ImportError as e:
  USE_NSML = False
  print('\nThis script will be ran on the local machine')

parser = argparse.ArgumentParser(description='Retinaface WiderFace Inference')
parser.add_argument('--network', default='efficientb4-3fpn', help='Backbone network mobile0.25 or resnet50 or mobilelight')
parser.add_argument('--weights', default='dongmin/efficientb4-3fpn_weights.pth')
parser.add_argument('--FPN', default='FPN', help='FPN, BiFPN')
parser.add_argument('--FPN_stack', default=3, type=int, help='number of FPN layers (only valid for BiFPN)')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--dataset_folder', default='taey16/storage/widerface/WIDER_val/images/', help='Validation dataset (Wider Face) directory')
parser.add_argument('--annotation_txt', default='dongmin/wider_val.txt')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--save_folder', default='dongmin/widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--image_save_folder', default='dongmin/widerface_evaluate/widerface_img/', type=str, help='Dir to save image results')
parser.add_argument('--device', default='cuda')
parser.add_argument('--vis_thres', default=0.0, type=float, help='visualization_threshold')
parser.add_argument('--pretrained_backbone', default='', help='pretrained pth path for backbone network')
args = parser.parse_args()

args.dataset_folder = os.path.join('/home', args.dataset_folder) if not USE_NSML else os.path.join(DATASET_PATH[1], args.dataset_folder)
args.weights = os.path.join('/home', args.weights) if not USE_NSML else os.path.join(DATASET_PATH[0], args.weights)
args.annotation_txt = os.path.join('/home', args.annotation_txt) if not USE_NSML else os.path.join(DATASET_PATH[0], args.annotation_txt)
args.save_folder = os.path.join('/home', args.save_folder) if not USE_NSML else os.path.join(DATASET_PATH[0], args.save_folder)
args.image_save_folder = os.path.join('/home', args.image_save_folder) if not USE_NSML else os.path.join(DATASET_PATH[0], args.image_save_folder)
args.device = torch.device(args.device)

def check_keys(model, pretrained_state_dict):
    pretrained_state_dict_new = dict()
    for key in pretrained_state_dict.keys():
        if 'fpn.body.p' in str(key):
            pretrained_state_dict_new[str(key).replace('body', 'FPNs.0')] = pretrained_state_dict[key]
        else:
            pretrained_state_dict_new[key] = pretrained_state_dict[key]

    ckpt_keys = set(pretrained_state_dict_new.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return pretrained_state_dict_new


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    pretrained_dict = check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # net and model
    net, cfg = load_network(args, phase='test')
    train_state_dict = torch.load(args.weights, map_location=args.device)

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
    train_state_dict = check_keys(net, train_state_dict)

    net.load_state_dict(train_state_dict)
    net.to(torch.device(args.device))
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.annotation_txt

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i in range(num_images):
        img_name = test_dataset[i]
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(args.device)
        scale = scale.to(args.device)

        _t['forward_pass'].tic()
        with torch.no_grad():
            loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width),mode='floor')
        priors = priorbox.forward()
        priors = priors.to(args.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(args.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        # print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
        
        if i % 20 == 0:
            print("[%4d/%4d] Finished" % (i+1, num_images))
        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            img_save_name = args.image_save_folder + img_name[:-4] + ".png"
            img_dirname = os.path.dirname(img_save_name)
            if not os.path.isdir(img_dirname):
                os.makedirs(img_dirname)
            cv2.imwrite(img_save_name, img_raw)

import os
import cv2
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

def visualize_batch_result(images, out, img_paths, cfg, global_cnt, rgb_mean = (104, 117, 123),
                        confidence_threshold = 0.05, top_k = 7500,
                        nms_threshold = 0.3, keep_top_k = 5000,
                        vis_thres = 0.0, save_dir = 'vis'):

    for batch in range(len(images)):
        img_path = img_paths[batch]
        img_raw = np.moveaxis(images[batch].cpu().numpy(), 0, -1).copy()
        img_raw = img_raw + rgb_mean
        img_raw = img_raw.astype(np.uint8)
        height, width, _ = img_raw.shape
        
        loc, conf, landms = [out[0][batch].cpu(), out[1][batch].cpu(), out[2][batch].cpu()]

        priorbox = PriorBox(cfg, image_size=(height, width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data, prior_data, cfg['variance'])
        boxes = boxes.cpu().numpy()
        conf = conf.softmax(dim=1)[:, 1]
        scores = conf.data.cpu().numpy()
        landms = decode_landm(landms.data, prior_data, cfg['variance'])
        landms = landms.cpu().numpy()

        # ignore low scores
        confidence_threshold = np.max(scores)
        inds = np.where(scores >= confidence_threshold)[0]
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

        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        n_pred = len(dets)
        img_w_annot = img_raw.copy()

        # show image
        for b in dets:
            if b[4] < vis_thres:
                continue

            b[[0, 2, 5, 7, 9, 11, 13]] *= width
            b[[1, 3, 6, 8, 10, 12, 14]] *= height
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_w_annot, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_w_annot, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_w_annot, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_w_annot, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_w_annot, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_w_annot, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_w_annot, (b[13], b[14]), 1, (255, 0, 0), 4)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result_img = np.zeros((height, width*2 + 10, 3))
        result_img[:, :width] = img_raw.copy()
        result_img[:, width+10:] = img_w_annot.copy()
        cv2.imwrite(os.path.join(save_dir, '%04d_%s_%s_result.png' % (global_cnt+1, *img_path.split(os.sep)[-2:])), result_img)
        global_cnt += 1
    
    return global_cnt
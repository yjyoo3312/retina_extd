import torch
import numpy as np

from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm

def get_max_pred_landmark(out, priors, cfg):
    loc, conf, landms = out
    prior_data = priors.data
    boxes = decode(loc.data, prior_data, cfg['variance'])

    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data, prior_data, cfg['variance']).cpu().numpy()

    inds = np.where(scores == np.max(scores))[0]

    box = boxes[inds]
    landm = landms[inds]
    score = scores[inds]

    return box, landm, score

def NME(out, targets, priors, cfg):
    nme_list = np.zeros(5)
    batch_size = len(targets)

    for batch in range(batch_size):
        box, landm, scores = get_max_pred_landmark([out[0][batch], out[1][batch], out[2][batch]], priors, cfg)
        pts = np.concatenate([box[0], landm[0]])
        pts_gt = targets[batch][0].cpu().detach().numpy()
        
        nme_list += compute_NME(pts, pts_gt)
    
    return nme_list / batch_size


def compute_NME(pts, pts_gt):
    # Normalize factor = sqrt(bbox width * bbox height)
    x_min, y_min, x_max, y_max = pts_gt[:4]
    box_size = np.sqrt((x_max-x_min)*(y_max-y_min))
    
    nme_list = []  # left_eye_center, right_eye_center, nose_tip, left_mouth_corner, right_mouth_corner
    for i in range(4, 14, 2):
        pred = pts[i:i+2]
        gt   = pts_gt[i:i+2]
        diff = (pred - gt)**2
        diff = np.sqrt(np.sum(diff))

        nme = diff / box_size
        nme_list.append(nme)
    
    return np.array(nme_list)
#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import random

import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy.io import loadmat
import argparse

import torch
import torch.utils.data as data

try:
  import nsml
  from nsml import DATASET_PATH, NSML_NFS_OUTPUT
  USE_NSML = True
  print('\nThis script will be ran on the NSML')
except ImportError as e:
  USE_NSML = False
  print('\nThis script will be ran on the local machine')

class AFLWFace():
    def __init__(self, index, name, mask, landmark, box):
        self.image_path = name
        self.face_id = index
        self.face_box = np.array([int(box[0]), int(box[2]), int(box[1]), int(box[3])])
        mask = np.expand_dims(mask, axis=1)
        landmark = landmark.copy()
        self.landmarks = np.concatenate((landmark, mask), axis=1)

def xy2wh(face_box):
    '''
    input : [x_min, y_min, x_max, y_max]
    output : [x_center, y_center, width, height]
    '''
    
    x_min, y_min, x_max, y_max = face_box
    
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    width = x_max - x_min
    height = y_max - y_min
    
    return np.array([x_center, y_center, width, height])

def xy2paddedxy(face_box, width_pad_ratio=0.15, height_pad_ratio=0.15):
    x_center, y_center, width, height = xy2wh(face_box)
    
    expand_width = width * (1.0 + width_pad_ratio)
    expand_height = height * (1.0 + height_pad_ratio)
    
    x_min_pad = int(x_center - expand_width //2)
    y_min_pad = int(y_center - expand_height//2)
    x_max_pad = int(x_center + expand_width //2)
    y_max_pad = int(y_center + expand_height//2)
    
    zero_pad_size = int(max([width * width_pad_ratio, height * height_pad_ratio]))
    
    return [x_min_pad, y_min_pad, x_max_pad, y_max_pad], zero_pad_size

def move_landmarks(landmarks, x_min_pad, y_min_pad):
    landmarks_moved = landmarks.copy()
    
    # x-coord
    landmarks_moved[range(0, len(landmarks), 2)] = landmarks_moved[range(0, len(landmarks), 2)] - x_min_pad
    # y-coord
    landmarks_moved[range(1, len(landmarks), 2)] = landmarks_moved[range(1, len(landmarks), 2)] - y_min_pad
    
    return landmarks_moved

def crop_face(aflwface, data_dir, width_pad_ratio=0.15, height_pad_ratio=0.15):
    img = cv2.imread(os.path.join(data_dir, aflwface.image_path))
    
    face_box = aflwface.face_box
    landmarks = aflwface.landmarks[:, :2].reshape(-1)
    
    [x_min_pad, y_min_pad, x_max_pad, y_max_pad], zero_pad = xy2paddedxy(face_box, width_pad_ratio, height_pad_ratio)
    img = np.pad(img, ((zero_pad, zero_pad), (zero_pad, zero_pad), (0, 0)), 'constant')
    
    cropped_img = img[y_min_pad + zero_pad : y_max_pad + zero_pad,
               x_min_pad + zero_pad : x_max_pad + zero_pad].copy()
    
    landmarks_moved = move_landmarks(landmarks, x_min_pad, y_min_pad)
    return cropped_img, landmarks_moved

def annot(img, landmarks):
    landmarks = np.array(landmarks).astype(int)
    img_annot = img.copy()
    
    for i in range(0, len(landmarks), 2):
        cv2.circle(img_annot, (landmarks[i], landmarks[i+1]), 1, (0, 0, 255), 4)
    
    return img_annot

def main(opt):
    mat_path = 'AFLWinfo_release.mat'
    aflwinfo = dict()
    mat = loadmat(os.path.join(opt.aflw_dir, mat_path))
    total_image = 24386

    # load train/test splits
    ra = np.squeeze(mat['ra']-1).tolist()
    aflwinfo['train-index'] = ra[:20000]
    aflwinfo['test-index'] = ra[20000:]
    aflwinfo['name-list'] = []

    # load name-list
    for i in range(total_image):
        name = mat['nameList'][i,0][0]
        aflwinfo['name-list'].append( name )

    aflwinfo['mask'] = mat['mask_new'].copy()
    aflwinfo['landmark'] = mat['data'].reshape((total_image, 2, 19))
    aflwinfo['landmark'] = np.transpose(aflwinfo['landmark'], (0,2,1))
    aflwinfo['box'] = mat['bbox'].copy()
    allfaces = []

    for i in range(total_image):
        face = AFLWFace(i, aflwinfo['name-list'][i], aflwinfo['mask'][i], aflwinfo['landmark'][i], aflwinfo['box'][i])
        allfaces.append( face )

    face_cnt_dict = dict()
    for face in tqdm(allfaces):
        img_path = face.image_path
        img_name = img_path.split(os.sep)[-1]
        subset = 'train' if face.face_id in aflwinfo['train-index'] else 'test'

        if img_path in face_cnt_dict:
            face_cnt_dict[img_path] += 1
        else:
            face_cnt_dict[img_path] = 1

        cropped_img, landmarks = crop_face(face, opt.aflw_dir, opt.width_pad_ratio, opt.height_pad_ratio)
        img_annoted = annot(cropped_img, landmarks)
        
        save_folder = os.path.join(opt.save_dir, subset, img_path.split(os.sep)[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        cv2.imwrite(os.path.join(save_folder, img_name.replace('.jpg', '_%d.png' % face_cnt_dict[img_path])), cropped_img)
        cv2.imwrite(os.path.join(save_folder, img_name.replace('.jpg', '_%d_annoted.jpg' % face_cnt_dict[img_path])), img_annoted)
        np.save(os.path.join(save_folder, img_name.replace('.jpg', '_%d_gt.npy' % face_cnt_dict[img_path])), landmarks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='')
    
    p.add_argument('--aflw_dir', type=str)
    p.add_argument('--save_dir', type=str)
    p.add_argument('--width_pad_ratio', default=0.15, type=float)
    p.add_argument('--height_pad_ratio', default=0.15, type=float)

    opt = p.parse_args()
    
    if USE_NSML:
        from nsml import DATASET_PATH
        print("NSML DATASET PATH : %s - NSML_NFS_OUTPUT : %s" % (DATASET_PATH, NSML_NFS_OUTPUT))

        opt.aflw_dir = os.path.join(DATASET_PATH, opt.aflw_dir)
        opt.save_dir = os.path.join(DATASET_PATH, opt.save_dir)
        
    main(opt)
    print("Done")
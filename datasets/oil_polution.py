# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------

import os
import copy
import random

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class OilPolution(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=6,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 hsv_aug=True,
                 mba_aug=True):

        super(OilPolution, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]
        self.hsv_aug = hsv_aug
        self.mba_aug = mba_aug
        self.files = self.read_files()

        self.label_mapping = {19: 0, 20: 1, 21: 2, 22: 3, 23: 4, 24: 5}
        self.class_weights = torch.FloatTensor([1.1000, 1.0009, 0.9000, 1.5000, 1.8000 , 1.9000]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                new_files = [{
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "aug": []
                }]
                if self.hsv_aug:
                    self.extendAugmentFile(new_files, 'HSV', 10)
                if self.mba_aug:
                    self.extendAugmentFile(new_files, 'MBA', 2)

                files.extend(new_files)
        return files
        
    def convert_label(self, label, inverse=False):
        temp = np.full_like(label, self.ignore_label)
        if inverse:
            for v, k in self.label_mapping.items():
                temp[label == k] = v
        else:
            for k, v in self.label_mapping.items():
                temp[label == k] = v
        return temp

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        # Our data augmentation
        if 'HSV' in item['aug']:
            gamma = random.uniform(0, 2.0)
            image = self.random_HSV_augment(image, label, gamma, self.label_mapping[24])
        if 'MBA' in item['aug']:
            alpha = random.uniform(0, 0.3)
            image = self.mixingBackgroundAugmentation(image, label, alpha, self.label_mapping[24])

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

    def random_HSV_augment(self, imgOrigin, imgSeg, gamma, target_label):
        '''
        Random specific label in HSV color space
        input
            imgOrigin: Original BGR image
            imgSeg: Gray-scale segmentation label
            gamma: A float to affect color change
            target_label: Label needs to be apply change
        output
            imgNew: The image after augmentation
        '''
        imgHSV = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2HSV)
        imgNew = imgOrigin.copy()

        imgHSV[:, :, 0][imgSeg == target_label] = (
            ((imgHSV[imgSeg == target_label][:, 0] / 180) ** gamma) * 180)
        imgNew = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)

        return imgNew

    def mixingBackgroundAugmentation(self, imgOrigin, imgSeg, alpha, target_label):
        '''
        introduce
            This function is a data augmentation method propose by
            LIGHT-WEIGHT MIXED STAGE PARTIAL NETWORK FOR SURVEILLANCE OBJECT DETECTION WITH BACKGROUND DATA AUGMENTATION
            https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506212
        input
            imgOrigin: Original BGR image
            imgSeg: Gray-scale segmentation label
            alpha: A float to affect color change
            target_label: Label needs to be apply change
        output
            imgNew: The images list after applying MBA
        '''
        imgNew = imgOrigin.copy()
        mask = np.zeros(imgNew.shape, np.uint8)
        mask[imgSeg == 24] = 255
        return cv2.addWeighted(imgNew, 1 - alpha, mask, alpha, 0)

    def extendAugmentFile(self, new_files, tag, n_aug):
        aug_files = []
        for file in new_files:
            for i in range(n_aug):
                new_file = copy.deepcopy(file)
                new_file['aug'].append(tag)
                aug_files.append(new_file)
        new_files.extend(aug_files)
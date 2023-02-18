# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# example command: python3 tools\custom.py --model-type 'pidnet-l' --model output\elanroad\best.pt --input samples\Cam5.mp4 --n-class 2 --visualize --show
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
from utils.dataloader import LoadImages
from utils.utils import time_synchronized

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(0, 0, 0),#[128, 192, 128], #(128, 64,128),
             [128, 64, 128], #(244, 35,232),
             [64, 64, 0], #( 70, 70, 70),
             [0, 192, 128], #(102,102,156),
             [192, 64, 0], #(190,153,153),
             [64, 192, 0], #(153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--model-type', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-l', type=str)
    parser.add_argument('--n-class', help='Number of categories', type=int, default=19)
    parser.add_argument('--model', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--input', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--visualize', help='Output results combined with semantic segmentation', action='store_true')     
    parser.add_argument('--show', help='Show output', action='store_true')     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def main(args):
    dataloader = LoadImages(args.input)
    vid_writer = vid_path = None
    
    model = models.pidnet.get_pred_model(args.model_type, args.n_class)
    model = load_pretrained(model, args.model).cuda()
    model.eval()
    with torch.no_grad():
        for img_path, imgOrigin, vid_cap in dataloader:
            sv_img = np.zeros_like(imgOrigin).astype(np.uint8)
            img = input_transform(imgOrigin)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            t1 = time_synchronized()
            pred = model(img)
            t2 = time_synchronized()
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            # Apply color
            for i in np.unique(pred):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            if args.visualize:
                sv_img = cv2.addWeighted(imgOrigin, 0.5, np.array(sv_img), 0.5, 0)
            if args.show:
                cv2.imshow("Result", sv_img)
                cv2.waitKey(1)
            
            # Save results
            sv_path, img_name = os.path.split(img_path)
            sv_path = os.path.join(sv_path, 'outputs')
            save_path = os.path.join(sv_path, img_name)
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            if dataloader.mode == 'image':
                cv2.imwrite(save_path, sv_img)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(sv_img)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')

    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()
            
if __name__ == '__main__':
    args = parse_args()
    main(args)
# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# example command: python tools/eval.py --cfg configs/cityscapes/pidnet_small_MBA_HSV_AUG.yaml TEST.MODEL_FILE output/cityscapes/pidnet_small_MBA_HSV_AUG/best.pt
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')      
   
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        hsv_aug=config.TEST.HSV_AUG,
                        mba_aug=config.TEST.MBA_AUG)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    
    start = timeit.default_timer()
    
    if ('test' in config.DATASET.TEST_SET) and ('city' in config.DATASET.DATASET):
        test(config, 
             test_dataset, 
             testloader, 
             model,
             sv_dir=final_output_dir)
        
    else:
        mean_IoU, IoU_array, pixel_acc, mean_acc, f1_scores, precisions, recalls = \
            testval(config, test_dataset, testloader, model)
    
        msg = f'MeanIoU: {mean_IoU:.4f}, Pixel_Acc: {pixel_acc:.4f}, ' +\
            f'Mean_Acc: {mean_acc:.4f}, Class IoU:{[f"{score:.4f}" for score in IoU_array]}, ' +\
            f'Precisions: {[f"{score:.3f}" for score in precisions]}, ' +\
            f'Recalls: {[f"{score:.3f}" for score in recalls]}, ' +\
            f'F1_scores: {[f"{score:.3f}" for score in f1_scores]}'
        logging.info(msg)
        print(msg)

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end-start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()

# Oil Polution Dataset with PIDNet
	
This is **not** the official repository for PIDNet ([PDF](https://arxiv.org/pdf/2206.02066)）

## Highlights
<p align="center">
  <img src="figs/cityscapes_score.jpg" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Comparison of inference speed and accuracy for real-time models on test set of Cityscapes.</span> 
</p>

* **Towards Real-time Applications**: PIDNet could be directly used for the real-time applications, such as autonomous vehicle and medical imaging.
* **A Novel Three-branch Network**: Addtional boundary branch is introduced to two-branch network to mimic the PID controller architecture and remedy the overshoot issue of previous models.
* **More Accurate and Faster**: PIDNet-S presents 78.6% mIOU with speed of 93.2 FPS on Cityscapes test set and 80.1% mIOU with speed of 153.7 FPS on CamVid test set. Also, PIDNet-L becomes the most accurate one (80.6% mIOU) among all the real-time networks for Cityscapes.

## Demos

A demo of the segmentation performance of our proposed PIDNets: Original video (left) and predictions of PIDNet-S (middle) and PIDNet-L (right)
<p align="center">
  <img src="figs/video1_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #1</span>
</p>

<p align="center">
  <img src="figs/video2_all.gif" alt="Cityscapes" width="800"/></br>
  <span align="center">Cityscapes Stuttgart demo video #2</span>
</p>

## Overview
<p align="center">
  <img src="figs/pidnet.jpg" alt="overview-of-our-method" width="800"/></br>
  <span align="center">An overview of the basic architecture of our proposed Proportional-Integral-Derivative Network (PIDNet). </span> 
</p>
P, I and D branches are responsiable for detail preservation, context embedding and boundary detection, respectively.

### Detailed Implementation
<p align="center">
  <img src="figs/pidnet_table.jpg" alt="overview-of-our-method" width="500"/></br>
  <span align="center">Instantiation of the PIDNet for semantic segmentation. </span> 
</p>
For operation, "OP, N, C" means operation OP with stride of N and the No. output channel is C; Output: output size given input size of 1024; mxRB: m residual basic blocks; 2xRBB: 2 residual bottleneck blocks; OP<sub>1</sub>\OP<sub>2</sub>: OP<sub>1</sub> is used for PIDNet-L while OP<sub>1</sub> is applied in PIDNet-M and PIDNet-S. (m,n,C) are scheduled to be (2,3,32), (2,3,64) and (3,4,64) for PIDNet-S, PIDNet-M and PIDNet-L, respectively.

## Models
For simple reproduction, here provided the ImageNet pretrained models.
Also, the finetuned models on Oil Polution are available for direct application in marine oil polution detection.

| Model | Links |
| :-: | :-: |
| ImageNet Pretrained | [PIDNet-S](https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing) |
| Finetuned Oil Polution | [PIDNet-S](https://ntutcc-my.sharepoint.com/:u:/g/personal/111598401_cc_ntut_edu_tw/EeCUhUsx0dVKgBW-zx5nJy4BpdP9SEf7JzLhyeCfyWIj6A?e=vxkusi) |

| Oil Polution Dataset | Links |
| :-: | :-: |
| Dataset | [Download](https://ntutcc-my.sharepoint.com/:u:/g/personal/111598401_cc_ntut_edu_tw/Ed_Ye-Y6osJJlyW1Vr8MNTABH9m9wPQ8i8hUdRBl70Gukw?e=k2YQ9L) |
| Config | [Download](https://ntutcc-my.sharepoint.com/:u:/g/personal/111598401_cc_ntut_edu_tw/EeVjZAVXEoNEkXnUPUJhicsBYOLhLmR2Mm8xffY5x3k2Cg?e=ojfJ1E) |

## Usage

### 0. Prepare the dataset

* Download the [OilPolution](https://ntutcc-my.sharepoint.com/:u:/g/personal/111598401_cc_ntut_edu_tw/Ed_Ye-Y6osJJlyW1Vr8MNTABH9m9wPQ8i8hUdRBl70Gukw?e=k2YQ9L)   datasets and [configuration](https://ntutcc-my.sharepoint.com/:u:/g/personal/111598401_cc_ntut_edu_tw/EeVjZAVXEoNEkXnUPUJhicsBYOLhLmR2Mm8xffY5x3k2Cg?e=ojfJ1E) files, unzip them and replace in `root` dir.
* Check if the paths contained in lists of `data/list` are correct for dataset images.

### 1. Training

* Download the [ImageNet pretrained models](https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing) and put them into `pretrained_models/imagenet/` dir.
* For example, train the PIDNet-S on OilPolution with batch size of 12 on 2 GPUs:
````bash
python tools/train.py --cfg configs/oilpolution/pidnet_small_MBS_HSV_AUG_S.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 6
````

### 2. Evaluation & Speed Measurement

* check [offical repository](https://github.com/XuJiacong/PIDNet)

### 4. Custom Inputs

* Put all your images in `samples/` and then run the command below using OilPolution pretrained PIDNet-S for image format of .jpg:
````bash
python tools/custom.py --a pidnet-s --p output/oilpolution/pidnet_small_MBA_HSV_AUG_S/best.pt --t .jpg --r samples\
````

## Acknowledgement

* Our implementation is based on [PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller](https://github.com/XuJiacong/PIDNet).
* Thanks for their nice contribution.


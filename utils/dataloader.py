# ------------------------------------------------------------------------------
# Modified based on https://github.com/WongKinYiu/yolov7
# ------------------------------------------------------------------------------

import numpy as np
import glob
import cv2
import os

class LoadImages:
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

    def __init__(self, path):
        if '*' in path:
            files = sorted(glob.glob(path, recursive=True))  # glob
        elif os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))  # dir
        elif os.path.isfile(path):
            files = [path]  # files
        else:
            raise Exception(f'ERROR: {path} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in self.img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in self.vid_formats]
        n_img, n_video = len(images), len(videos)

        self.files = images + videos
        self.n_file = n_img + n_video  # number of files
        self.video_flag = [False] * n_img + [True] * n_video
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.n_file > 0, f'No images or videos found in {path}. ' \
                            f'Supported formats are:\nimages: {self.img_formats}\nvideos: {self.vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.n_file:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.n_file:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.n_file} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
            assert img is not None, 'Image Not Found ' + path
            
        return path, img, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.n_file  # number of files
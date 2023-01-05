import glob
import random
import os

img_dirs = glob.glob('../../oilpolution/Img/*.jpg')
img_dirs.extend(glob.glob('../../oilpolution/Img/*.png'))
random.shuffle(img_dirs)
n_data = len(img_dirs)

train = []
val = []
for i, img_dir in enumerate(img_dirs):
    _, img_name = os.path.split(img_dir)
    final_img_dir = os.path.join('oilpolution', 'Img', img_name)
    seg_dir = final_img_dir.replace('Img', 'Seg')
    seg_dir = seg_dir.replace('jpg', 'png')
    
    if i < n_data * 0.6:
        train.append(f'{final_img_dir}\t{seg_dir}\n')
    elif i < n_data * 0.8:
        val.append(f'{final_img_dir}\t{seg_dir}\n')
        
with open('train.lst', 'w') as f:
    f.writelines(train)
with open('val.lst', 'w') as f:
    f.writelines(val)
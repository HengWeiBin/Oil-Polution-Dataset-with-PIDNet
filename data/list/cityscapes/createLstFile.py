import argparse
import os
import random

def main(args):
    _ , tail = os.path.split(args.listFileDir)
    filename = tail.split('.')[0]

    pairs = None
    with open(args.listFileDir, 'r') as f:
        pairs = f.readlines()
        
    total_data_amount = len(pairs)
    train = []
    val = []
    for i in range(total_data_amount):
        img_dir, _ = pairs[i].split('\t')
        _, tail_img_dir = os.path.split(img_dir)
        
        if tail_img_dir.startswith('A'):
            val.append(pairs[i])
        else:
            train.append(pairs[i])
            
    with open(f'{filename}_train.lst', 'w') as f:
        f.writelines(train)
    with open(f'{filename}_val.lst', 'w') as f:
        f.writelines(val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--listFileDir", type=str, default="../../cityscapes/MBA_HSV_AUG_Slist.lst")
    args = parser.parse_args()

    main(args)
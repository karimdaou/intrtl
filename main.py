import numpy as np
import cv2
import pickle as pkl
import argparse
import os
import random
from classes import *
from functions import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', help="# of augmented images per file", type=int)
    parser.add_argument('-p', '--prob', nargs='+', help="Probability of applying the transform", type=float)
    parser.add_argument('-a', '--angle', nargs='+', help='Range from which a random angle is picked', type=int)
    parser.add_argument('-b', '--brightness', nargs='+', help='Range for changing brightness', type=int)
    parser.add_argument('-v', action='store_true', help="Enables visualization of augmented images and bboxes.")

    args = parser.parse_args()

# specify where to load images from and where to save the augmented ones
load_from_path = os.path.join(os.getcwd(), 'data')
write_to_path = os.path.join(os.getcwd(), 'augmented_data')

# check if the target directory exists, if not create it
if not os.path.exists(write_to_path):
    os.mkdir(write_to_path)
    print("Directory ", write_to_path, " Created ")

with open(os.path.join(load_from_path, 'imgs_info.pickle'), 'rb') as bb_cords:
    bboxes = pkl.load(bb_cords)

# get file names of images from the given dataset
images = []
# r=root, d=directories, f = files
for r, d, f in os.walk(load_from_path):
    for file in f:
        if '.jpg' in file:
            images.append(file)

output_data = []

for i in range(args.num):
    for f in images:
        image = cv2.imread(os.path.join(load_from_path, f))
        boxes = None
        aug = DualTransform([RandomBrightness(args.brightness), RandomRotate(args.angle)], args.prob)

        for data in bboxes:
            if data['img_name'] == f:
                boxes = np.array(data['coors'])
                image, boxes = aug(image, boxes)

                file_name = str(i) + '_' + f
                output_data.append(save_augmented_data(write_to_path, image, boxes, file_name, args.v))

# Finally saving the info about augmented bounding boxes into a pickle file              
with open(os.path.join(write_to_path, 'aug_imgs_info.pickle'), 'wb') as output_file:
    pkl.dump(output_data, output_file)

import numpy as np
import cv2
import random
from functions import *


class RandomRotate(object):
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box coordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, angle=10):
        self.angle = angle

        if type(self.angle) == list:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes=None):

        angle = random.uniform(*self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        img = rotate_im(img, angle)

        if bboxes is not None:
            corners = get_corners(bboxes)

            corners = np.hstack((corners, bboxes[:, 4:]))

            corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

            new_bbox = get_enclosing_box(corners)

            scale_factor_x = img.shape[1] / w

            scale_factor_y = img.shape[0] / h

            new_bbox[:, :4] = new_bbox / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

            bboxes = new_bbox

        img = cv2.resize(img, (w, h))

        return img, bboxes


class RandomBrightness(object):
    """Randomly changes the brightness of an image    
    
    Parameters
    ----------
    value: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`value`, `value`). If **tuple**,
        the `value` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Brighter(Darker) image in the numpy format of shape `HxWxC`
        
    numpy.ndarray
        Untouched bounding box coordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, value=10):
        self.value = value

        if type(self.value) == list:
            assert len(self.value) == 2, "Invalid range"

        else:
            self.value = (-self.value, self.value)

    def __call__(self, img, bboxes=None):

        value = random.uniform(*self.value)

        img = change_brightness(img, value)

        return img, bboxes


class DualTransform(object):
    """Initialise Dual Trasformation object
    
    Apply a Sequence of transformations to the images/boxes.
    
    Parameters
    ----------
    augemnetations : list 
        List containing Transformation Objects in Sequence they are to be 
        applied
    
    probs : int or list 
        If **int**, the probability with which each of the transformation will 
        be applied. If **list**, the length must be equal to *augmentations*. 
        Each element of this list is the probability with which each 
        corresponding transformation is applied
    
Returns
    -------
    
    numpy.ndaaray
        Augmented image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box coordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, augmentations, probs=1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list and len(self.probs) > 1:
                prob = self.probs[i]
            else:
                prob = self.probs[0]

            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes

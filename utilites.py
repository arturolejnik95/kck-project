from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte


def remove_not_silver(img):
    difference = 20
    new_img = img.copy()
    for rowindex, row in enumerate(img):
        for index, (b, g, r) in enumerate(row):		
            if abs(int(b) - int(g)) >= difference or abs(int(b) - int(r)) >= difference or abs(int(r) - int(g)) >= difference:
                new_img[rowindex][index] = (0,0,0)
    return new_img
	
def remove_not_gold(img):
    difference = 20
    new_img = img.copy()
    for rowindex, row in enumerate(img):
        for index, (b, g, r) in enumerate(row):		
            if b > 150 or r < 150 or g < 140 or g > 200:
                new_img[rowindex][index] = (0,0,0)
    return new_img
	
	
def apply_brightness_contrast(input_img, brightness, contrast):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
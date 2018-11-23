from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte

def avgColor(img):
    blue = 0
    green = 0
    red = 0
    i = 0
    for row in img:
        for (b, g, r) in row:
            if(b != 0 and g != 0 and r != 0):
                blue = blue + b
                green = green + g
                red = red + r
                i = i + 1
    blue = blue/i
    green = green/i
    red = red/i
    return blue, green, red

def resizing(img, size):
    if img.shape[0] < img.shape[1]:
        d = size/ img.shape[1]
        dim = (size, int(img.shape[0] * d))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        d = size / img.shape[0]
        dim = (int(img.shape[1] * d), size)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def remove_not_silver(img):
    difference = 20
    white_limit = 240
    black_limit = 20
    new_img = img.copy()
    for rowindex, row in enumerate(img):
        for index, (b, g, r) in enumerate(row):		
            if abs(int(b) - int(g)) >= difference or abs(int(b) - int(r)) >= difference or abs(int(r) - int(g)) >= difference or (b > white_limit and r > white_limit and g > white_limit) or (b < black_limit and r < black_limit and g < black_limit):
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
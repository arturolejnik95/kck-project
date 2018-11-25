from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

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
    if i > 0:
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
    img = apply_brightness_contrast(img, 40, 0)
    #cv2.imshow('apply_brightness_contrast', img)
    difference = 40
    white_limit = 250
    black_limit = 5
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

def coinsColor(img):
    surArea = img.shape[0] * img.shape[1]
	
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(image_hsv)
    saturation.fill(0)
    hue.fill(255)
    image_hsv = cv2.merge([hue, saturation, value])
    out = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)	

    cv2.imshow('coinsColor result', out)

    return False

def compareContoursArtur(cnt1, cnt2):
    contours = []
    if len(cnt1) > 0:
        for c1 in cnt1:
            contours.append(c1)
    if len(cnt2) > 0:    
        for c2 in cnt2:
            intersect = False
            for c1 in cnt1:
                area, intersection = cv2.intersectConvexConvex(c1,c2)
                print('Intersect: {}'.format(area))
                if area > 0:
                    intersect = True
            if not intersect:
                contours.append(c2)
    return contours
	
def coinsColor(img):
    surArea = img.shape[0] * img.shape[1]
	
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(image_hsv)
    saturation.fill(0)
    hue.fill(255)
    image_hsv = cv2.merge([hue, saturation, value])
    out = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)	

    cv2.imshow('coinsColor result', out)

    return False
	
def compareContours(cnt1, cnt2):
    distance1 = 0
    distance2 = 0
    distance = False
	
    for point in cnt2:
        dist = cv2.pointPolygonTest(cnt1,(point[0][0], point[0][1]),False)
        distance1 += dist
        if dist == 0:
            distance1 += 1	
			
    for point in cnt1:
        dist = cv2.pointPolygonTest(cnt2,(point[0][0], point[0][1]),False)
        distance2 += dist
        if dist == 0:
            distance2 += 1
		
    if distance1 > 0:
        distance = True
        
		
    if distance2 > 0:
        distance = True		
		
    print("compareContours", distance1, len(cnt2), distance2, len(cnt1))
    return distance
	
def addNewContours(new, offContours, image):
    offContoursCopy = offContours    
    if len(new) != 0:
        for cnt1 in new:
            if len(offContours) == 0:
                offContoursCopy = new
                break
            flag = False
            for index, cnt2 in enumerate(offContours):
                #print("index", index)
                if compareContours(cnt2, cnt1):
                    flag = True
                    print("index", index)
            if not flag:
                offContoursCopy.append(cnt1)
                print("added")
                #if offContoursCopy is not None:
                #    cv2.drawContours(image, offContoursCopy, -1, (0,255,0), 3)
                #    cv2.imshow("new", image)
                #    cv2.waitKey(0)
    return offContoursCopy
'''
    for row in coin_image:
        for (b, g, r) in row:  
            if b != 0 and g != 0 and r != 0:
                b_color.append(b)
                g_color.append(g)
                r_color.append(r)
                print("color:", (b, g ,r))
'''
def coinValue(coin_image, maska):
    hsv = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(hsv)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([hsv],[i],maska,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
		
    plt.show()
    return 0
	
def cropContour(image, contour):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image_g) # Create mask where white is what we want, black otherwise
    #cv2.drawContours(mask, [contour], 0, (0,255,0), 3) # Draw filled contour in mask
    cv2.drawContours(mask, [contour], -1, 255, 1) # Draw filled contour in mask
    out = np.zeros_like(image) # Extract out the object and place into output image
    cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
    cv2.imshow('mask', mask)
    out[mask == 255] = image[mask == 255]

 
    # Show the output image
    cv2.imshow('crop', out)
    coinValue(out, mask)	
    cv2.waitKey(0)
    return 0

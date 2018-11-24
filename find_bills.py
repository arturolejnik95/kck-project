from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte
from utilites import apply_brightness_contrast


def findBillsArtur(img, coins):
    surArea = img.shape[0] * img.shape[1]
    contours = []
    
    # do the laplacian filtering
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.uint8(img)
    imgResult = sharp - imgLaplacian
    
    # convert back to 8bits gray scale
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')    
    
    #contrast and brightness
    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 150:
        imgResult = apply_brightness_contrast(imgResult, 0, 127)
    elif 120 < np.mean(gray) < 150:
        imgResult = apply_brightness_contrast(imgResult, -70, 127)
    else:
        imgResult = apply_brightness_contrast(imgResult, 80, 127)
	    
    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 180:
        gray = 255 - gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel1 = np.ones((2, 2), dtype=np.uint8)
    kernel2 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1,1]], dtype=np.uint8)
    kernel3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel1)
    _, binary = cv2.threshold(binary,0,255,cv2.THRESH_BINARY)
    binary = cv2.erode(binary,kernel3,iterations = 2)
    if np.mean(binary) > 128:
        binary = 255 - binary
        
    
    cv2.imshow('findBillsBinary', binary)
    
    _, cont, _ = cv2.findContours(binary , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        
        if len(approx) > 3 and 0.05 * surArea < area < 0.40 * surArea:
            peri2 = 6*np.sqrt(area/2)
            if 1.1 > peri/peri2 > 0.9:
                contours.append(approx)
    print("findBillsArtur found: ", len(contours))
    return contours
	
def findBillsD(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]
    #convert to HSV color scheme
    flat_object_resized_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split HSV to three chanels
    hue, saturation, value = cv2.split(flat_object_resized_hsv)
    # threshold to find the contour

    retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	#wypełnienie dziur
    thresholded_open = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (7,7))
    thresholded_close = cv2.morphologyEx(thresholded_open, cv2.MORPH_CLOSE, (7,7))
	
    _, cont, _ = cv2.findContours(thresholded_close , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        # approximate the contour
        # These methods are used to approximate the polygonal curves of a contour. 
        # In order to approximate a contour, you need to supply your level of approximation precision. 
        # In this case, we use 2% of the perimeter of the contour. The precision is an important value to consider. 
        # If you intend on applying this code to your own projects, you’ll likely have to play around with the precision value.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4 and 0.05 * surArea < area < 0.95 * surArea:
            contours.append(approx)
    print("findBillsD found: ", len(contours))
    return contours
	
def findBillsA(img):
    surArea = img.shape[0] * img.shape[1]
    contours = []
    
    # do the laplacian filtering
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    sharp = np.uint8(img)
    imgResult = sharp - imgLaplacian
    
    # convert back to 8bits gray scale
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')    
    
    #contrast and brightness
    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 150:
        imgResult = apply_brightness_contrast(imgResult, 0, 127)
    elif 120 < np.mean(gray) < 150:
        imgResult = apply_brightness_contrast(imgResult, -70, 127)
    else:
        imgResult = apply_brightness_contrast(imgResult, 80, 127)
    
    
    #transform image
    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 180:
        gray = 255 - gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel1 = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=np.uint8)
    kernel2 = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=np.uint8)
    kernel3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    binary = cv2.dilate(binary,kernel3,iterations = 2)
    _, binary = cv2.threshold(binary,1,128,cv2.THRESH_BINARY_INV)
    binary = cv2.erode(binary,kernel3,iterations = 1)
    if np.mean(binary) > 64:
        binary = 128 - binary
        
    #watershed    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    _, dist = cv2.threshold(dist, 0.05, 1.0, cv2.THRESH_BINARY)

    kernel2 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel2)
    dist_8u = dist.astype('uint8')    

    _, cont, _ = cv2.findContours(dist_8u , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        # approximate the contour
        # These methods are used to approximate the polygonal curves of a contour. 
        # In order to approximate a contour, you need to supply your level of approximation precision. 
        # In this case, we use 2% of the perimeter of the contour. The precision is an important value to consider. 
        # If you intend on applying this code to your own projects, you’ll likely have to play around with the precision value.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4 and 0.05 * surArea < area < 0.95 * surArea:
            contours.append(approx)	
    print("findBillsA found: ", len(contours))			
    return contours
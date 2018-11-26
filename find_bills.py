from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte
from utilites import apply_brightness_contrast
from utilites import compareContoursArtur


def findBillsArtur(img, coins):
    surArea = img.shape[0] * img.shape[1]
    contours = []
    
    blur = cv2.GaussianBlur(img,(15,15),0)
    cv2.imshow('Blur',blur)
    contrast = apply_brightness_contrast(blur, 0, 85)
    cv2.imshow('Contrast',contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        contrast = 255 - contrast
    
    kernel1 = np.array([[-1,-1,-1],[-1,30,-1],[-1,-1,-1]])
    sharp = cv2.filter2D(contrast,-1,kernel1)
    cv2.imshow('Sharp',sharp)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  
    cv2.imshow('Binary',binary)
    
    if np.mean(binary) > 127:
        binary = 255 - binary
    cv2.imshow('Binary2',binary)
    
    cont1 = []
    _, cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        
        if 7 > len(approx) > 3 and 0.03 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                cont1.append(approx)
                
    coins2 = []
    if coins is not None:            
        for c in coins:
            inside = False
            for cnt in cont1:
                area, intersection = cv2.intersectConvexConvex(c,cnt)
                if area > 0:
                    inside = True
            if not inside:
                cv2.drawContours(binary, [c], 0, (0, 0, 0), 2)
                coins2.append(c)
    
    kernel2 = np.ones((5,5), dtype=np.uint8)
    kernel3 = np.ones((8,8), dtype=np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel3)
    binary = cv2.dilate(binary,kernel2,iterations=6)
    binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, kernel3)
    binary = cv2.erode(binary,kernel2,iterations=3)

    if np.mean(binary) > 150:
        binary = 255 - binary
        
    cont2 = []
    _, cont, _ = cv2.findContours(binary , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        
        if 7 > len(approx) > 3 and 0.05 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                cont2.append(approx)
                
    contours = compareContoursArtur(cont1,cont2)

    cv2.imshow('Binary2',binary)
    print("findBillsArtur found: ", len(contours))
    return contours, coins2



def findBillsBright(img, coins):
    surArea = img.shape[0] * img.shape[1]
    contours = []
    contrast = apply_brightness_contrast(img, 0, 85)
    gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 127:
        contrast = 255 - contrast
    gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY) 
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    _, cont, _ = cv2.findContours(binary , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]

    for cnt in cont:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        
        if 7 > len(approx) > 3 and 0.05 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                contours.append(approx)

    coins2 = []
    if coins is not None:            
        for c in coins:
            inside = False
            for cnt in contours:
                area, intersection = cv2.intersectConvexConvex(c,cnt)
                if area > 0:
                    inside = True
            if not inside:
                coins2.append(c)
    
    return contours, coins2

def findBillsContrast(img, coins):
    surArea = img.shape[0] * img.shape[1]
    contours = []
    contrast = apply_brightness_contrast(img, 0, 127)
    contrast = 255 - contrast
    cv2.imshow('Contrast', contrast)
    gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary', binary)
    _, cont, _ = cv2.findContours(binary , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]
    for cnt in cont:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        
        if 7 > len(approx) > 3 and 0.05 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                contours.append(approx)

    coins2 = []
    if coins is not None:            
        for c in coins:
            inside = False
            for cnt in contours:
                area, intersection = cv2.intersectConvexConvex(c,cnt)
                if area > 0:
                    inside = True
            if not inside:
                coins2.append(c)
    
    print("findBillsContrast found: ", len(contours))
    return contours, coins2
    
	
def findBillsD(img, coins):
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
        if 7 > len(approx) > 3 and 0.05 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                contours.append(approx)
    print("findBillsD found: ", len(contours))
    coins2 = []
    if coins is not None:            
        for c in coins:
            inside = False
            for cnt in contours:
                area, intersection = cv2.intersectConvexConvex(c,cnt)
                if area > 0:
                    inside = True
            if not inside:
                coins2.append(c)
    
    return contours, coins2
	
def findBillsA(img, coins):
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
        if 7 > len(approx) > 3 and 0.05 * surArea < area < 0.30 * surArea:
            peri2 = 6*np.sqrt(area*1.25/2)
            if 1.15 > peri/peri2 > 0.85:
                contours.append(approx)
    print("findBillsA found: ", len(contours))			
    coins2 = []
    if coins is not None:            
        for c in coins:
            inside = False
            for cnt in contours:
                area, intersection = cv2.intersectConvexConvex(c,cnt)
                if area > 0:
                    inside = True
            if not inside:
                coins2.append(c)
    
    return contours, coins2

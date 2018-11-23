from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte
from utilites import apply_brightness_contrast
from utilites import remove_not_silver

def findCoinsAdaptiveThresholding(img):
    contours = []
	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
	
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    cv2.imshow("gray_blur", gray_blur)
    
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    cv2.imshow("adaptive", thresh)
	
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cont_img = closing.copy()
    _, contours,_ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 4000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0,255,0), 2)
	
	
    cv2.imshow('final result', img)
    return contours

def findCoinsBright(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]

    new_image = np.zeros(img.shape, img.dtype)

    alpha = 2 # Simple contrast control
    beta = -600   # Simple brightness control
	

    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)	
	
    flat_object_resized_hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(flat_object_resized_hsv)
	
    flat_object_resized_hsv = resizing(flat_object_resized_hsv, 600)
    cv2.imshow('flat_object_resized_hsv', flat_object_resized_hsv)
    retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		
    cv2.imshow('thresholded', thresholded)	
	 
	
    _, cont, _ = cv2.findContours(thresholded , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
	
	
    new_image = resizing(new_image, 600)	
    cv2.imshow('Original Image', img)
    cv2.imshow('New Image', new_image)

	
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
	
    cv2.imshow('contours', img)	
    return contours	

def findSilverCoins(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]

    img = remove_not_silver(img)
    cv2.imshow("img", img)		
    #convert to HSV color scheme
    flat_object_resized_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split HSV to three chanels
    hue, saturation, value = cv2.split(flat_object_resized_hsv)
    # threshold to find the contour
	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	
	#wypełnienie dziur
    thresholded_open = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (7,7))
    thresholded_close = cv2.morphologyEx(thresholded_open, cv2.MORPH_CLOSE, (7,7))
	
    thresholded_close = np.invert(thresholded_close)
	
	
    cv2.imshow('thresholded_close', thresholded_close)
	
    _, cont, _ = cv2.findContours(thresholded_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if len(approx) > 5:
            contours.append(approx) 	
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow("silverConturs", img)
    return contours
	
def findCoinsArtur(img):
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
    
    D = ndimage.distance_transform_edt(dist_8u)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=dist_8u)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=dist_8u)
    
    #contours
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(dist_8u.shape, dtype="uint8")
        mask[labels == label] = 255
        
        _, cont, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cont:
            area = cv2.contourArea(cnt)
            if 0.002*surArea < area < 0.05*surArea:
                (x, y), rad = cv2.minEnclosingCircle(cnt)
                rad = rad * 0.95
                area2 = np.pi*pow(rad,2)
                if 0.002*surArea < area2 < 0.05*surArea and area/area2 > 0.6:
                    contours.append((int(x),int(y),int(rad)))
                    cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)   
    return contours
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
from utilites import resizing
from utilites import avgColor
import colorsys


def findHoughCircles(image):
    contours = [] 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = image.copy()
	
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
    if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	    circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	    for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	    #cv2.imshow("output", np.hstack([image, output]))
	    cv2.waitKey(0)
    print("findHoughCircles found: ", len(contours))
    return contours
	
def findCoinsAdaptiveThresholding(img):
    contours = []
	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    #cv2.imshow('gray result', gray)	
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    #cv2.imshow("gray_blur", gray_blur)
    #cv2.imshow('gray_blur result', gray_blur)    
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

	
    if np.mean(thresh) > 128:
        thresholded = np.invert(thresh)
    #cv2.imshow('thresh result', thresh)
	
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cont_img = closing.copy()
    _, cont,_ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
    print(len(cont))	
    surArea = img.shape[0] * img.shape[1]	
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if 0.002*surArea < area < 0.05*surArea:
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            rad = rad * 0.95
            area2 = np.pi*pow(rad,2)
            if 0.002*surArea < area2 < 0.2*surArea and area/area2 > 0.6:
                contours.append(cnt)
                cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)
	
	
    #cv2.imshow('findCoinsAdaptiveThresholding result', img)
    print("findCoinsAdaptiveThresholding found: ", len(contours))
    return contours


	
def findCoinsBright(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]

    contrast = apply_brightness_contrast(img, -50, 100)
    #cv2.imshow('contrast', contrast)	 
    flat_object_resized_hsv = cv2.cvtColor(contrast, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(flat_object_resized_hsv)
	
    #cv2.imshow('value', value)
    retval, thresholded = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	
    if np.mean(thresholded) > 128:
        thresholded = np.invert(thresholded)	
    
    #cv2.imshow('thresholded', thresholded)
	
    _, cont, _ = cv2.findContours(thresholded , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
	
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if 0.002*surArea < area < 0.05*surArea:
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            rad = rad * 0.95
            area2 = np.pi*pow(rad,2)
            if 0.002*surArea < area2 < 0.2*surArea and area/area2 > 0.6:
                contours.append(cnt)
                cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)
				
    #cv2.imshow('Original Image', img)
    cv2.imshow('findCoinsBright result', img)

	
    print("findCoinsBright found: ", len(contours))	
    return contours
	
def findSilverCoins(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]

    img = remove_not_silver(img)
    #cv2.imshow("remove_not_silver", img)		

	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    # threshold to find the contour
    retval, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	
	#wypełnienie dziur
    thresholded_open = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, (7,7))
    thresholded_close = cv2.morphologyEx(thresholded_open, cv2.MORPH_CLOSE, (7,7))
	
    if np.mean(thresholded_close) > 128:
        thresholded_close = np.invert(thresholded_close)
    #cv2.imshow('thresholded_close', thresholded_close)
	
    _, cont, _ = cv2.findContours(thresholded_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]
	
    for cnt in cont:
        area = cv2.contourArea(cnt)
        if 0.002*surArea < area < 0.05*surArea:
            (x, y), rad = cv2.minEnclosingCircle(cnt)
            rad = rad * 0.95
            area2 = np.pi*pow(rad,2)
            if 0.002*surArea < area2 < 0.2*surArea and area/area2 > 0.6:
                contours.append(cnt)
                cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)
    #cv2.imshow("draw_cirlces", img)
    print("findSilverCoins found: ", len(contours))
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
                    contours.append(cnt)
                    cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2) 
    cv2.imshow("findCoinsArtur", img)					
    print("findCoinsArtur found: ", len(contours))					
    return contours
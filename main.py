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
        
    
    cv2.imshow('1', binary)
    cv2.waitKey(0)
    
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
    return contours
	
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
'''
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",img)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(img)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)
'''
def findCoinsBright(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]

    new_image = np.zeros(img.shape, img.dtype)

    alpha = 2 # Simple contrast control
    beta = -600   # Simple brightness control
	

    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)	
	
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
    return contours

def coinsValue(img, coins):
    for c, (x, y, rad) in enumerate(coins):
        rad2zl = int(rad * 12.50/21.00) #Stosunek promienia wewnetrznego do calej monety - 2 zlote
        rad5zl = int(rad * 16.00/24.00) #Stosunek promienia wewnetrznego do calej monety - 5 zlotych
        
        crop_img = img[y-rad:y+rad, x-rad:x+rad].copy()
        mask = np.zeros((crop_img.shape[0], crop_img.shape[1]), dtype = np.uint8)
        cv2.circle(mask, (rad, rad), rad, (255, 255, 255), -1, 8, 0)
        crop_img[mask[:,:] == 0] = 0
        
        crop_img2 = img[y-rad2zl:y+rad2zl, x-rad2zl:x+rad2zl].copy()
        mask2 = np.zeros((crop_img2.shape[0], crop_img2.shape[1]), dtype = np.uint8)
        cv2.circle(mask2, (rad2zl, rad2zl), rad2zl, (255, 255, 255), -1, 8, 0)
        crop_img2[mask2[:,:] == 0] = 0
        
        crop_img22 = crop_img.copy()
        cv2.circle(crop_img22, (rad, rad), rad2zl, (0,0,0), -1, 8, 0)
        
        crop_img5 = img[y-rad5zl:y+rad5zl, x-rad5zl:x+rad5zl].copy()
        mask5 = np.zeros((crop_img5.shape[0], crop_img5.shape[1]), dtype = np.uint8)
        cv2.circle(mask5, (rad5zl, rad5zl), rad5zl, (255, 255, 255), -1, 8, 0)
        crop_img5[mask5[:,:] == 0] = 0
        
        crop_img55 = crop_img.copy()
        cv2.circle(crop_img55, (rad, rad), rad5zl, (0,0,0), -1, 8, 0)
        
        blue, green, red = avgColor(crop_img)
        blue2, green2, red2 = avgColor(crop_img2)
        blue22, green22, red22 = avgColor(crop_img22)
        blue5, green5, red5 = avgColor(crop_img5)
        blue55, green55, red55 = avgColor(crop_img55)
        #print(blue, green, red)
        #print(blue2, green2, red2)
        #print(blue22, green22, red22)
        #print(blue5, green5, red5)
        #print(blue55, green55, red55)
        #print("")
        #cv2.imshow('1', crop_img22)
        #cv2.waitKey(0)
        
def billsValue(img, bills):
    return 0

name = 'money.jpg'
image = cv2.imread(name)
image = resizing(image, 500)
image2 = image.copy()

coins = findCoinsArtur(image)
bills = findBillsArtur(image, coins)
#if coins is not None:
    #coinsValue(image, coins)
if bills is not None:
    billsValue(image, bills)
    cv2.drawContours(image, bills, -1, (0,255,0), 3)

cv2.imshow(name, image)
cv2.waitKey(0)

bills2 = findBillsD(image2)
coinsAdaptive = findCoinsBright(image2)

cv2.imshow(name, image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

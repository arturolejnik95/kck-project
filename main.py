from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

def avgColor(img):
    blue = 0
    green = 0
    red = 0
    i = 0
    for row in img:
        for (b, g, r)  in row:
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

def findCoins(img):
    contours = []
    surArea = img.shape[0] * img.shape[1]
    
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    _, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel1 = np.ones((7,7), np.uint8)
    kernel2 = np.ones((12,12), np.uint8)
    kernel3 = np.ones((7,7), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    thresh = cv2.erode(thresh, kernel3, iterations=1)
    cv_image = thresh.copy()
    
    cv2.imshow('1', thresh)
    cv2.waitKey(0)
    
    D = ndimage.distance_transform_edt(cv_image)
    
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=cv_image)
    
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=cv_image)
    
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        
        
        _, cont, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cont:
            approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
            if len(approx) > 3:
                area = cv2.contourArea(cnt)
                if 0.001*surArea < area < 0.03*surArea:
                    (x, y), rad = cv2.minEnclosingCircle(cnt)
                    rad = rad * 0.95
                    area2 = np.pi*pow(rad,2)
                    if area/area2 > 0.7:
                        contours.append((int(x),int(y),int(rad)))
                        cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)
    return contours

def findBills(img, coins):
    contours = []
    surArea = img.shape[0] * img.shape[1]
    
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    _, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    for c in coins:
        x, y, rad = c
        cv2.circle(thresh, (int(x), int(y)), int(rad + 10), (0, 0, 0), -1)
    
    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((25,25), np.uint8)
    kernel3 = np.ones((10,10), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations=4)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel3)

    cv2.imshow('2', thresh)    
    cv2.waitKey(0)
    
    _, cont, _ = cv2.findContours(thresh , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
image2 = image.copy

coins = findCoins(image)
bills = findBills(image, coins)
if coins is not None:
    coinsValue(image, coins)
if bills is not None:
    billsValue(image, bills)
    cv2.drawContours(image, bills, -1, (0,255,0), 3)

cv2.imshow(name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

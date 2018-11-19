import cv2
import numpy as np
from skimage import img_as_ubyte

def avgColor(img):
    blue = 0
    green = 0
    red = 0
    i = 0
    for row in img:
        for (b, g, r)git  in row:
            if(b != 0 and g != 0 and r != 0):
                blue = blue + b
                green = green + g
                red = red + r
                i = i + 1
    blue = blue/i
    green = green/i
    red = red/i
    return blue, green, red

def findCoins(img,surArea):
    contours = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    closing = img_as_ubyte(cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2))
    
    _, cont, _ = cv2.findContours(closing , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cont:
        approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
        if len(approx) > 3:
            area = cv2.contourArea(cnt)
            if 0.01*surArea < area < 0.04*surArea:
                (x, y), rad = cv2.minEnclosingCircle(cnt)
                rad=rad*0.9
                area2 = np.pi*pow(rad,2)
                if area/area2 > 0.7:
                    contours.append((int(x),int(y),int(rad)))
    return contours
	
def resize(img, width=None, height=None, interpolation = cv2.INTER_AREA):
    global ratio
    w, h, _ = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height/h
        width = int(w*ratio)
        print(width)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width/w
        height = int(h*ratio)
        print(height)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
		
def findBills(img,surArea):
    contours = []
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
	
    for c in cnts:
        # approximate the contour
        # These methods are used to approximate the polygonal curves of a contour. 
        # In order to approximate a contour, you need to supply your level of approximation precision. 
        # In this case, we use 2% of the perimeter of the contour. The precision is an important value to consider. 
        # If you intend on applying this code to your own projects, you’ll likely have to play around with the precision value.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
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
image = cv2.imread(name, 0)
rows, cols = image.shape
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
image = cv2.imread(name)

coins = findCoins(image, nrows*ncols)
bills = findBills(resize(image, height=600), nrows*ncols)
if coins is not None:
    coinsValue(image, coins)
if bills is not None:
    billsValue(image, bills)
    cv2.drawContours(resize(image, height=600), bills, -1, (0,255,0), 3)

cv2.imshow(name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

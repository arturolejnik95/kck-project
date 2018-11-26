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

    #cv2.imshow('coinsColor result', out)

    return False

def billsValueHSV(original, bill):
    h1 = 0
    v1 = 0
    s1 = 0
    il = 0
    hsv = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
    for row in hsv:
        for h, s, v in row:
            h1 = h1 + h
            v1 = v1 + v
            s1 = s1 + s
            il = il + 1
    if il > 0:
        h1 = h1/il
        s1 = s1/il
        v1 = v1/il   
        kernel1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        img = original.copy()
        mask = np.zeros(img.shape, dtype = np.uint8)
        cv2.drawContours(mask, [bill], 0, (255, 255, 255), -1)
        img[mask[:,:] == 0] = 0
        hsv2 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h2 = 0
        s2 = 0
        v2 = 0
        il2 = 0        
        for row in hsv2:
            for h, s, v in row:
                if v > 50:
                    h2 = h2 + h
                    s2 = s2 + s
                    v2 = v2 + v
                    il2 = il2 + 1
        if il2 > 0:
            h2 = h2/il2
            s2 = s2/il2
            v2 = v2/il2
            val = h2/il2
            if val < 20:
                values = "20zl"
            else:
                values = "50zl"
        else:
            values = "Blad"
    else:
        values = "Blad"
    return values

def billsValue(original,bills):
    values = []    
    for c, bill in enumerate(bills):
        img = original.copy()
        mask = np.zeros(img.shape, dtype = np.uint8)
        cv2.drawContours(mask, [bill], 0, (255, 255, 255), -1)
        img[mask[:,:] == 0] = 0

        b, g, r = avgColor(img)
        if abs(b - r) <= 10 or b > r or r <= 160 or b <= 135:
            values.append("50zl")
        elif r >= 225 or (r >= 210 and abs(r - b) >= 50) or (r >= 200 and abs(r - b) >= 50) or r - b >= 65:
            values.append("20zl")
        else:
            values.append(billsValueHSV(original,bill))
    return values

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
                #print('Intersect: {}'.format(area))
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

    #cv2.imshow('coinsColor result', out)

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
    #print("distance1", distance1)	
    if distance1 > -10:
        distance = True
    #print("distance2", distance2)
    if distance2 > -10:
        distance = True		
		
    #print("compareContours", distance1, len(cnt2), distance2, len(cnt1))
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
                    #print("index", index)
            if not flag:
                offContoursCopy.append(cnt1)
                #print("added")
                #if offContoursCopy is not None:
                #    cv2.drawContours(image, offContoursCopy, -1, (0,255,0), 3)
                #    cv2.imshow("new", image)
                #    cv2.waitKey(0)
    return offContoursCopy

def watersheding(binary, contours):
    contours2 = []
    surArea = binary.shape[0] * binary.shape[1]
    if len(contours) > 0:
        for c in contours:
            contours2.append(c)
    #watershed    
    surArea = binary.shape[0] * binary.shape[1]
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    _, dist = cv2.threshold(dist, 0.05, 1.0, cv2.THRESH_BINARY)

    #kernel2 = np.ones((3,3), dtype=np.uint8)
    #dist = cv2.dilate(dist, kernel2)
    dist_8u = dist.astype('uint8')    

    #cv2.imshow("findCoinsDist", dist)
    
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
                if 0.05*surArea < area2 < 0.04*surArea and area/area2 > 0.6:
                    contours2.append(cnt)
    return contours2
	
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
                    #print("index", index)
            if not flag:
                offContoursCopy.append(cnt1)
                #print("added")
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
                #print("color:", (b, g ,r))
'''
def coinValue(coin_image, maska, center, radius):
    hsv = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)	
    hue, saturation, value = cv2.split(hsv)
    #histr = cv2.calcHist([hsv],[2],maska,[256],[0,256])
    x = int(center[1])
    y = int(center[0])
    #print("xy", x, y)
    srodek = coin_image[x][y]
    #print("srodek", srodek)
    x2 = int(x - (radius * 1 / 2))
    y2 = int(y - (radius * 1 / 2))
    skraj = coin_image[x2][y2]
    #print("skraj", skraj)
    #w1 = 
    #print("w1", w1)
    if(int(srodek[0]) - int(skraj[0]) > 10):
        value = "2zl"
    elif(int(skraj[0]) - int(srodek[0]) > 15):
        value = "5zl"
    else:
        value = "20gr"
    #color = ('b','g','r')
    #for i,col in enumerate(color):
        #histr = cv2.calcHist([hsv],[i],maska,[256],[0,256])
        #print(i, np.max(histr))
		
	
    return value
	
def compareRadiuses(maxRadius, radiuses):
    values = []
    flag2 = 0
    flag5 = 0
    for radius in radiuses:
        if maxRadius != radius:
            #print(maxRadius / radius)
            if maxRadius / radius < 1.05:
                values.append("biggest")
            #5zł / 2zl
            if maxRadius / radius < 1.11 + (0.05):
                values.append("2zl")
                flag5 += 1
            #2zł / 20gr
            elif maxRadius / radius < 1.16 + (0.1):
                values.append("20gr")
                flag2 += 1
            #5zł / 20gr
            elif maxRadius / radius < 1.30 + (0.05):
                values.append("20gr")
                flag5 += 1
            else:
                values.append("blank")
        else:
            values.append("biggest") 
    for i, value in enumerate(values):
        if value == "biggest":
            if flag2 > flag5:
                values[i] = "2zl"
            elif flag5 > flag2:
                values[i] = "5zl"
            elif flag5 == 0:
                values[i] = "20gr"
            else:
                values[i] = "blank"
        			
    return values
def getCenter(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    return (cX, cY)
	
def getRadius(contour):
    (x, y), rad = cv2.minEnclosingCircle(contour)
    return rad
	
def getValueFromColor(img, cnt):
    value = cropContour(img, cnt)
    return value
	
def cropContour(image, contour):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image_g) # Create mask where white is what we want, black otherwise
    #cv2.drawContours(mask, [contour], 0, (0,255,0), 3) # Draw filled contour in mask
    cv2.drawContours(mask, [contour], -1, 255, 1) # Draw filled contour in mask
    out = np.zeros_like(image) # Extract out the object and place into output image
    cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
    #cv2.imshow('mask', mask)
    out[mask == 255] = image[mask == 255]

 
    # Show the output image
    #cv2.imshow('crop', out)
    value = coinValue(out, mask, getCenter(contour), getRadius(contour))	
    #cv2.waitKey(0)
    return value

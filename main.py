from __future__ import print_function
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from skimage import img_as_ubyte

from find_coins import findCoinsArtur
from find_coins import findSilverCoins
from find_coins import findCoinsBright
from find_coins import findCoinsAdaptiveThresholding
from find_coins import findHoughCircles

from find_bills import findBillsArtur
from find_bills import findBillsD
from find_bills import findBillsA

from utilites import remove_not_silver
from utilites import remove_not_gold
from utilites import resizing
from utilites import avgColor
from utilites import apply_brightness_contrast
from utilites import compareContours
from utilites import addNewContours

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
        
def billsValue(img, bills, dwadziescia, piecdziesiat):
    for c, bill in enumerate(bills):        
        kernel1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharp = cv2.filter2D(img,-1,kernel1)
        contrast = apply_brightness_contrast(sharp, 0, 127)
        blur = cv2.GaussianBlur(contrast,(15,15),0)
        
        mask = np.zeros(blur.shape, dtype = np.uint8)
        cv2.drawContours(mask, [bill], 0, (255, 255, 255), -1)
        blur[mask[:,:] == 0] = 0
        
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        i = 0
        j = 0
        hsum = 0
        k = 0
        for row in hsv:
            for h, s, v in row:
                if s < 50 and v > 205:
                    s = 0
                    v = 0
                    blur[i,j,0] = 0
                    blur[i,j,1] = 0
                    blur[i,j,2] = 0
                if s > 50 and v > 50 and v < 205:
                    hsum = hsum + h
                    k = k + 1                    
                j = j + 1
            i = i + 1
            j = 0
        b, g, r = avgColor(blur)
        if k > 0:
            val = hsum/k
            print('{}'.format(val))
            if val < 15:
                dwadziescia = dwadziescia + 1
            else:
                piecdziesiat = piecdziesiat + 1
    return dwadziescia, piecdziesiat

'''
    if len(silver) != 0:
        for cnt1 in silver:
            if len(offContours) == 0:
                offContours = silver
                break
            for cnt2 in offContours:
                if not compareContours(cnt2, cnt1):
                    offContours.append(cnt1)
'''
	
names = [0] * 143
numbers = ["%03d" % i for i in range(1,27)]
for i, number in enumerate(numbers):
    if i < -1:
        continue
    names[i] = "picture_" + numbers[i] + ".jpg"
    image = cv2.imread("nasze/" + names[i])
    image = resizing(image, 500)
    image2 = image.copy()
	
    offContours = []

    #findHoughCircles(image)

    silver = findSilverCoins(image)
    #bills2 = findBillsD(image2)

    #coinsBright = findCoinsBright(image2)
    #coinsAdaptive = findCoinsAdaptiveThresholding(image2)
	
    #coins = findCoinsArtur(image)
    #bills = findBillsArtur(image, coins)
    #bills3 = findBillsA(image2)
	
    #h = findHoughCircles(image)
	
    #offContours = coins
	

    #offContours = addNewContours(coinsAdaptive, offContours, image)
    offContours = addNewContours(silver, offContours, image)
    #offContours = addNewContours(coinsBright, offContours, image)
    #offContours = addNewContours(bills, offContours, image)
    #offContours = addNewContours(bills2, offContours, image)
    #offContours = addNewContours(bills3, offContours, image)


	
    if offContours is not None:
        cv2.drawContours(image, offContours, -1, (0,255,0), 3)
    cv2.imshow(names[i], image)
    cv2.waitKey(0)
        #cv2.circle(image, (int(contour[0]), int(contour[1])), int(contour[2]), (0, 255, 0), 2)

    #if bills is not None:
        #dwadziescia, piecdziesiat = billsValue(image2, bills, 0, 0)
        #cv2.drawContours(image, bills, -1, (0,255,0), 3)
		
    #if bills2 is not None:
        #cv2.drawContours(image, bills2, -1, (0,255,0), 3)		

	
    #if bills3 is not None:
        #cv2.drawContours(image, bills3, -1, (0,255,0), 3)	
		
    
     

    #print('Dwadziescia: {}'.format(dwadziescia))
    #print('Piecdziesiat: {}'.format(piecdziesiat))
	

    cv2.destroyAllWindows()
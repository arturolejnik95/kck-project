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
from find_coins import findCoinsProg
from find_coins import findCoinsGaussian1
from find_coins import findCoinsGaussian2
from find_coins import findCoinsContrast


from find_bills import findBillsArtur
from find_bills import findBillsD
from find_bills import findBillsA
from find_bills import findBillsBright
from find_bills import findBillsContrast

from utilites import remove_not_silver
from utilites import remove_not_gold
from utilites import resizing
from utilites import avgColor
from utilites import apply_brightness_contrast
from utilites import compareContours
from utilites import addNewContours
from utilites import compareContoursArtur
from utilites import coinsColor
from utilites import cropContour
from utilites import getRadius
from utilites import compareRadiuses
from utilites import getCenter
from utilites import getMax

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
        
def billsValue1(original, bill):
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
            values.append(billsValue1(original,bill))
    return values

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
numbers = ["%03d" % i for i in range(1,28)]
for i, number in enumerate(numbers):
    if i < -1:
        continue
    names[i] = "picture_" + numbers[i] + ".jpg"
    image = cv2.imread("nasze/" + names[i])
    image = resizing(image, 500)
    #cv2.imshow(names[i], image)
    image0 = image.copy()
    image11 = image.copy()
    image12 = image.copy()
    image13 = image.copy()
    image14 = image.copy()
    image15 = image.copy()
    image16 = image.copy()
    image17 = image.copy()
    image18 = image.copy()
    image21 = image.copy()
    image22 = image.copy()
    image23 = image.copy()
    image24 = image.copy()
    image25 = image.copy()
    image3 = image.copy()
    imageCoinsValues = image.copy()
	

    offContours1 = []
    allCoins = []
    allBills = []
    allContours = []
    coinsColor(image0)


    silver = findSilverCoins(image11)
    coinsBright = findCoinsBright(image12)
    coinsAdaptive = findCoinsAdaptiveThresholding(image13)
    coins = findCoinsArtur(image14)
    coinsProg = findCoinsProg(image15)
    coinsGauss1 = findCoinsGaussian1(image16)
    coinsGauss2 = findCoinsGaussian2(image17)
    coinsContrast = findCoinsContrast(image18)

    
    allCoins.append(silver)
    allCoins.append(coinsBright)
    allCoins.append(coinsAdaptive)

    allCoins.append(coinsProg)
    allCoins.append(coinsGauss2)
    allCoins.append(coinsGauss1)
    allCoins.append(coinsContrast)
	
    allCoins.append(coins)

    offContours1 = []
	
    for cnt in allCoins:
         offContours1 = addNewContours(cnt, offContours1, image)

	
    bills1, offContours2 = findBillsBright(image21, offContours1)
    bills2, offContours3 = findBillsArtur(image22, offContours2)
    bills3, offContours4 = findBillsD(image23, offContours3)
    bills4, offContours5 = findBillsA(image24, offContours4)
    bills5, offContours6 = findBillsContrast(image25, offContours5)
    
    bills6 = compareContoursArtur(bills1, bills2)
    bills7 = compareContoursArtur(bills6, bills3)
    bills8 = compareContoursArtur(bills7, bills4)
    allBills = compareContoursArtur(bills8, bills5)

    radiuses = []
    if offContours6 is not None:
        cv2.drawContours(image3, offContours6, -1, (0,255,0), 3)
    for cnt in offContours6:
        print("radius:", getRadius(cnt))	
        radiuses.append(getRadius(cnt))
        #cropContour(imageCoinsValues, cnt)
    values = compareRadiuses(getMax(radiuses), radiuses)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for b in allBills:
        cv2.drawContours(image3, [b], 0, (0,255,0), 3) 

    for index, cnt in enumerate(offContours6):
        cv2.putText(image3,values[index], getCenter(cnt), font, 0.5,(0,0,255), 2)

    values2 = billsValue(image.copy(),allBills)
    if len(values2) > 0:
        for index, cnt in enumerate(allBills):
            cv2.putText(image3,values2[index], getCenter(cnt), font, 0.5,(0,0,255), 2)
    #for value in values:
        #print("value:", value)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow(names[i], image3)
    cv2.waitKey(0)

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

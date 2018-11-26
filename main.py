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
from utilites import getValueFromColor
from utilites import billsValue
#from utilites import getMax
	
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
	
    imageCopy = image.copy()
	
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
        cv2.drawContours(image3, offContours6, -1, (0,255,0), 2)
    for cnt in offContours6:
        #print("radius:", getRadius(cnt))	
        radiuses.append(getRadius(cnt))
        #cropContour(imageCoinsValues, cnt)
    if len(radiuses) > 0:
        values = compareRadiuses(np.max(radiuses), radiuses)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for b in allBills:
        cv2.drawContours(image3, [b], 0, (0,255,0), 2) 

    for index, cnt in enumerate(offContours6):
        if(values[index] == "blank"):
            values[index] = getValueFromColor(imageCopy, cnt)
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

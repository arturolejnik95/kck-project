import numpy as np
import cv2

def findCoins(img):
    contours = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    circles = cv2.HoughCircles(closing,cv2.HOUGH_GRADIENT,2,20,param1=450,param2=60,minRadius=0,maxRadius=0)
    for i in circles[0,:]:
        if 5000 < i[2]*i[2]*np.pi < 15000:
            # draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
            cir = (i[0], i[1], i[2])
            contours.append(cir)
    return contours

def findBills(img):
    contours = []
    
    return contours
    
image = cv2.imread('money.png')
coins = findCoins(image)
bills = findBills(image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

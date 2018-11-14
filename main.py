import cv2
import numpy as np
from skimage import img_as_ubyte


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
                    cv2.circle(img, (int(x), int(y)), int(rad), (0, 255, 0), 2)
                    cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 3)
    return contours

def findBills(img,surArea):
    contours = []
    
    return contours
    
image = cv2.imread('money.jpg',0)
rows, cols = image.shape
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
image = cv2.imread('money.jpg')
coins = findCoins(image, nrows*ncols)
bills = findBills(image, nrows*ncols)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

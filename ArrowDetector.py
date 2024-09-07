import cv2 as cv
import numpy as np
import math
image=cv.imread("arrow1.jpg")
img = cv.resize(image, (500, 500),cv.INTER_CUBIC)
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)


def redMasking(hsv):
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(mask1, mask2)
    return red_mask

def procImage(img): 
    blur = cv.GaussianBlur(img,(5,5),1)
    edges= cv.Canny(blur,200,200)   
    dialate = cv.dilate(edges,(5,5),iterations=3)
    eroded = cv.erode(dialate,(5,5),iterations=2)
    finalImg = eroded
    return finalImg

def fContour(contours): 
    max_contour = max(contours, key=cv.contourArea)
    return max_contour

def Fangle(contour):
    [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.001)    
    # extracting scalar values from the numpy kyuki nuopy vx and vy ko array ke form mai return karta hai
    vx = vx[0]
    vy = vy[0]
    x=x[0]
    y=y[0]
    angle_with_vertical = math.atan2(vx, vy)
    angle_with_vertical_degrees = math.degrees(angle_with_vertical)   
    return angle_with_vertical_degrees, vx, vy, x, y

redMask=redMasking(hsv)
finalImg=procImage(redMask)
ret,threshold_img=cv.threshold(finalImg,150,250,cv.THRESH_BINARY)
contours,herr=cv.findContours(threshold_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contour=fContour(contours)
image_main = img.copy()
cv.drawContours(image_main, [contour], -1, (93,170,40), 3)
angle,vx,vy,x,y=Fangle(contour)

#showing line
rows, cols = img.shape[:2]  
if vx != 0:
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
else:
    lefty = righty = int(y)

cv.line(image_main, (cols - 1, righty), (0, lefty), (0, 0, 0), 2)
cv.imshow('Fitted Line', image_main)
print("angle with vertical",angle)
cv.waitKey(0)
cv.destroyAllWindows()

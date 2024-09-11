import cv2 as cv
import numpy as np
import math
image=cv.imread("arrow1R.jpg")
hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)


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
def calculateAngle(pt1, pt2, pt3):
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle
def fContour(contours): 
    max_contour = max(contours, key=cv.contourArea)
    return max_contour
def findTip(contour):
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    min_angle = 360
    tip = None

    for i in range(len(approx)):
        prev_point = approx[i - 1][0]
        curr_point = approx[i][0]
        next_point = approx[(i + 1) % len(approx)][0]

        angle = calculateAngle(prev_point, curr_point, next_point)

        if angle < min_angle:
            min_angle = angle
            tip = curr_point

    return tip
def findCentroid(contour):
    M = cv.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) 
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return (cx, cy)
def Fangle(tip,centriod):
    x1, y1 = tip
    x2, y2 = centriod

    angle_with_vertical = math.atan2(x2-x1,y2-y1)
    angle_with_vertical_degrees = math.degrees(angle_with_vertical)   
    return angle_with_vertical_degrees

redMask=redMasking(hsv)
finalImg=procImage(redMask)
ret,threshold_img=cv.threshold(finalImg,150,250,cv.THRESH_BINARY)
contours,herr=cv.findContours(threshold_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contour=fContour(contours)
tip=findTip(contour)
centriod=findCentroid(contour)
image_main = image.copy()
cv.drawContours(image_main, [contour], -1, (93,170,40), 3)
angle=Fangle(tip,centriod)
rows, cols = image.shape[:2]  
image_main = image.copy()
cv.drawContours(image_main, [contour], -1, (93, 170, 40), 3)
cv.circle(image_main, tuple(tip), 8, (255, 0, 0), -1) 
cv.circle(image_main, centriod, 8, (0, 255, 0), -1)  
cv.line(image_main, tuple(tip), centriod, (0, 0, 0), 2)
cv.imshow('Arrow Analysis with Centroid', image_main)
print(Fangle(tip,centriod))
cv.waitKey(0)
cv.destroyAllWindows()
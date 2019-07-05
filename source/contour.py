import cv2
import imutils
import numpy as np
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
def detect(image):
    #image = cv2.imread('sleeve.jpg')
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    edged = cv2.Canny(gray, 1, 50)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    #cv2.imwrite('edged.jpg',edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0]
    #print(len(cnts))
    cnts = sorted(cnts, key=cv2.contourArea)

    epsilon = 0.0000000000000000000001*cv2.arcLength(cnts[-1],True)
    approx =  cv2.approxPolyDP(cnts[-1],epsilon,True)
    cv2.drawContours(image, [approx], 0, (0,255,0),3)

    box=cv2.minAreaRect(cnts[-1])
    box=cv2.cv.BoxPoints(approx) if imutils.is_cv2() else cv2.boxPoints(box)
    box=np.array(box, dtype="int")
    box=perspective.order_points(box)
    cv2.drawContours(image,approx,-1,(0,255,0),8)
    cv2.drawContours(image, [box.astype("int")], -1, (0,0,255),3)
    def midpoint(ptA, ptB):
    	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    for (x, y) in box:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)
                 
                    # compute the midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-righ and bottom-right
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)
                 
                    # draw the midpoints on the image
                    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                 
                    # draw lines between the midpoints
                    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                        (255, 0, 255), 2)
                    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                        (255, 0, 255), 2)
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    #cv2.imwrite('box.jpg',image)
    #print(dA)
    #print(dB)
    (tl, tr, br, bl) = box
    '''print(tl)
                print(tr)
                print(br)
                print(bl)'''
    return br[1] ,dA, dB


from source.user import  userPreprocess, catPreprocess
from source.contour import detect
import cv2
import os
import sys
from imutils import perspective
from imutils import contours
import numpy as np


grabcutOutput=cv2.imread(sys.argv[1])
catInst = catPreprocess(grabcutOutput)
floodOut = catInst.edgeDetect()
cropFlood = catInst.cropImg(floodOut)
processInst = userPreprocess(cropFlood)
processInst.cropImg()
processOut = processInst.removeTurds()
processInst.segImage(processOut)
LU, RU = processInst.getSegLines()
leftArmUser = processInst.armSegment(processOut,'left')
im_gray = cv2.cvtColor(leftArmUser, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]


width = 768
height=1024
pic = cv2.resize(im_bw, (width, height))
#cv2.imwrite('sleeve.jpg', pic)

lowestpoint,dA,dB=detect(pic)
#print(dA*dB)
#print(dA)
#print(dB)
area=dA*dB

if(lowestpoint>200 and lowestpoint<600 and area>10000):
    print('half-sleeves')
elif(lowestpoint>=600):
    print('full-sleeves')
else:
    print('sleeveless')


#cv2.imwrite('Video.jpg', image)
#cv2.imwrite('leftArmUser.png',pic)


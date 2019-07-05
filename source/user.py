#!/usr/bin/python

import cv2
import numpy as np
from scipy import ndimage
import Queue


class userPreprocess:
	def __init__(self,userImage):
		self.img = userImage
		self.leftSegLine = 0
		self.rightSegLine = 0
		self.Top = 0
		self.Bottom = 0
		self.Left = 0
		self.Right = 0
	
	def cropImg(self):
		labels, numLabels = ndimage.label(self.img)
		fragments = ndimage.find_objects(labels)
		self.Top = fragments[0][0].start
		self.Bottom = fragments[0][0].stop
		self.Left = fragments[0][1].start
		self.Right = fragments[0][1].stop
		Area = 0
		for slices in fragments:
			segArea = (slices[0].stop - slices[0].start)*(slices[1].stop - slices[1].start)
			if segArea >= Area:
				Area = segArea
				self.Top = slices[0].start
				self.Bottom = slices[0].stop
				self.Left = slices[1].start
				self.Right = slices[1].stop
		self.img = self.img[self.Top:self.Bottom, self.Left:self.Right]

	def returnUserBox(self):
		return [self.Top,self.Bottom,self.Left,self.Right]

	def removeTurds(self):
		grayImg = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		height = grayImg.shape[0]
		width = grayImg.shape[1]
		for i in xrange(height):
			if grayImg[i][width/2]:
				initPt = i
		#initPt += 1
		turdsOut = np.zeros((height,width,3))
		queue = Queue.Queue()
		visited = np.zeros((height,width))
		queue.put([initPt,width/2])
		while not queue.empty():
			a,b = queue.get()
			turdsOut[a][b] = self.img[a][b]
			if a>0 and b>0 and visited[a-1][b-1]==0 and grayImg[a-1][b-1] != 0:
				queue.put([a-1,b-1])
				visited[a-1][b-1]=1
			if a>0 and visited[a-1][b]==0 and grayImg[a-1][b] != 0:
				queue.put([a-1,b])
				visited[a-1][b]=1
			if b>0 and visited[a][b-1]==0 and grayImg[a][b-1] != 0:
				queue.put([a,b-1])
				visited[a][b-1]=1
			if a>0 and b<(width-1) and visited[a-1][b+1]==0 and grayImg[a-1][b+1] != 0:
				queue.put([a-1,b+1])
				visited[a-1][b+1]=1
			if b<(width-1) and visited[a][b+1]==0 and grayImg[a][b+1] != 0:
				queue.put([a,b+1])
				visited[a][b+1]=1
			if a<(height-1) and b<(width-1) and visited[a+1][b+1]==0 and grayImg[a+1][b+1] != 0:
				queue.put([a+1,b+1])
				visited[a+1][b+1]=1
			if a<(height-1) and visited[a+1][b]==0 and grayImg[a+1][b] != 0:
				queue.put([a+1,b])
				visited[a+1][b]=1
			if a<(height-1) and b>0 and visited[a+1][b-1]==0 and grayImg[a+1][b-1] != 0:
				queue.put([a+1,b-1])
				visited[a+1][b-1]=1

		#cv2.imwrite("debug/turdsOut.jpg",turdsOut)
		return np.uint8(turdsOut)
	
	def segImage(self,cropOut):
		grayUserImg = cv2.cvtColor(cropOut,cv2.COLOR_BGR2GRAY)
		i = grayUserImg.shape[0]/2
		for j in xrange(0, grayUserImg.shape[1]):
			if grayUserImg[i][j] != grayUserImg[0][0]:
				self.leftSegLine = j

				break

		for j in xrange(grayUserImg.shape[1]-1,-1,-1):
			if grayUserImg[i][j] != grayUserImg[0][0] :
				self.rightSegLine = j
				break 

		LPrev = self.leftSegLine
		RPrev = self.rightSegLine

		check = 0

		self.leftSegLine = 0
		self.rightSegLine = 0

		prevI = 0
		start = -1
		for i in xrange(grayUserImg.shape[0]/2, -1, -1):
			cnt=start
			for j in xrange(grayUserImg.shape[1]/2, -1, -1):
				if (cnt == 1) and (grayUserImg[i][j] != 0):
					cnt = 2
					start = 0
					prevI = i
					break
				if (cnt == start) and (grayUserImg[i][j] == 0):
					cnt = 1
			if (cnt == 1) and (start != -1):
				break


		for j in xrange(grayUserImg.shape[1]/2, -1, -1):
			if grayUserImg[prevI+1][j] == 0:
				prevJ1 = j
				break

		for j in xrange(grayUserImg.shape[1]/2, -1, -1):
			if grayUserImg[prevI][j] == 0 :
				prevJ2 = j
				break

		self.leftSegLine = min(prevJ1, prevJ2)

		prevI = 0
		start = -1
		for i in xrange(grayUserImg.shape[0]/2, -1, -1):
			cnt=start
			for j in xrange(grayUserImg.shape[1]/2, grayUserImg.shape[1]):
				if (cnt == 1) and (grayUserImg[i][j] != 0):
					cnt = 2
					start = 0
					prevI = i
					break
				if (cnt == start) and (grayUserImg[i][j] == 0):
					cnt = 1
			if cnt == 1 and start != -1:
				break


		for j in xrange(grayUserImg.shape[1]/2, grayUserImg.shape[1]):
			if grayUserImg[prevI+1][j] == 0 :
				prevJ1 = j
				break

		for j in xrange(grayUserImg.shape[1]/2, grayUserImg.shape[1]):
			if grayUserImg[prevI][j] == 0:
				prevJ2 = j
				break

		self.rightSegLine = min(prevJ1, prevJ2)


		if(abs(grayUserImg.shape[1]/2-self.leftSegLine)<abs(LPrev-self.leftSegLine)):
			self.leftSegLine = LPrev
		if(abs(grayUserImg.shape[1]/2-self.rightSegLine)<abs(RPrev-self.rightSegLine)):
			self.rightSegLine = RPrev
		if(abs(LPrev/2-self.leftSegLine)<abs(LPrev-self.leftSegLine)):
			self.leftSegLine = LPrev
		if(abs(RPrev + (grayUserImg.shape[1]-RPrev)/2 - self.rightSegLine)<abs(RPrev-self.rightSegLine)):
			self.rightSegLine = RPrev

	def getSegLines(self):
		return self.leftSegLine,self.rightSegLine

	def armSegment(self,img,leftOrRight):
		if leftOrRight == 'left':
			segImg = img.copy()
			segImg[:,self.leftSegLine:] = 0
			return self.armRemTurds(segImg,leftOrRight)
		else:
			segImg = img.copy()
			segImg[:,0:self.rightSegLine-1] = 0
			return self.armRemTurds(segImg,leftOrRight)

	def armRemTurds(self,img,leftOrRight):
		grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		height = img.shape[0]
		width = img.shape[1]
		queue = Queue.Queue()
		if leftOrRight == 'left':
			for i in xrange(img.shape[0]):
				if grayImg[i][self.leftSegLine-1] != 0:
					initPt = i
					break
			queue.put([initPt,self.leftSegLine-1])

		else:
			for i in xrange(img.shape[0]):
				if grayImg[i][self.rightSegLine+1] != 0:
					initPt = i
					break
			queue.put([initPt,self.rightSegLine+1])

		turdsOut = np.zeros((height,width,3))
		visited = np.zeros((height,width))

		while not queue.empty():
			a,b = queue.get()
			turdsOut[a][b] = img[a][b]
			if a>0 and b>0 and visited[a-1][b-1]==0 and grayImg[a-1][b-1] != 0:
				queue.put([a-1,b-1])
				visited[a-1][b-1]=1
			if a>0 and visited[a-1][b]==0 and grayImg[a-1][b] != 0:
				queue.put([a-1,b])
				visited[a-1][b]=1
			if b>0 and visited[a][b-1]==0 and grayImg[a][b-1] != 0:
				queue.put([a,b-1])
				visited[a][b-1]=1
			if a>0 and b<(width-1) and visited[a-1][b+1]==0 and grayImg[a-1][b+1] != 0:
				queue.put([a-1,b+1])
				visited[a-1][b+1]=1
			if b<(width-1) and visited[a][b+1]==0 and grayImg[a][b+1] != 0:
				queue.put([a,b+1])
				visited[a][b+1]=1
			if a<(height-1) and b<(width-1) and visited[a+1][b+1]==0 and grayImg[a+1][b+1] != 0:
				queue.put([a+1,b+1])
				visited[a+1][b+1]=1
			if a<(height-1) and visited[a+1][b]==0 and grayImg[a+1][b] != 0:
				queue.put([a+1,b])
				visited[a+1][b]=1
			if a<(height-1) and b>0 and visited[a+1][b-1]==0 and grayImg[a+1][b-1] != 0:
				queue.put([a+1,b-1])
				visited[a+1][b-1]=1

		#cv2.imwrite("debug/turdsOut.jpg",turdsOut)
		return np.uint8(turdsOut)

class catPreprocess:
	
	def __init__(self,img):
		self.img = img
		self.leftSegLine = 0
		self.rightSegLine = 0

	def edgeDetect(self,threshold=175):
		edgeImg = cv2.Canny(self.img,10,threshold)
		kernel = np.ones((3,3),np.uint8)
		grayDilated = cv2.dilate(edgeImg,kernel)
		kernel = np.ones((2,2),np.uint8)
		grayErode = cv2.erode(grayDilated,kernel)
		floodOut = self.img
		height,width = grayErode.shape[:2]
		queue = Queue.Queue()
		ref = grayErode[0][0]
		visited = np.zeros((height,width))
		queue.put([0,0])
		queue.put([height-1,width-1])
		queue.put([0, width-1])
		queue.put([height-1,0])
		queue.put([0, width/2])
		queue.put([height-1, width/2])
		while not queue.empty():
			a,b = queue.get()
			floodOut[a][b] = [0,0,0]
			if a>0 and b>0 and visited[a-1][b-1]==0:
				if grayErode[a-1][b-1] == ref:
					queue.put([a-1,b-1])
					visited[a-1][b-1]=1
				else:
					floodOut[a-1][b-1] = [0,0,0]
			if a>0 and visited[a-1][b]==0:
				if grayErode[a-1][b] == ref:
					queue.put([a-1,b])
					visited[a-1][b]=1
				else:
					floodOut[a-1][b] = [0,0,0]
			if b>0 and visited[a][b-1]==0:
				if grayErode[a][b-1] == ref:
					queue.put([a,b-1])
					visited[a][b-1]=1
				else:
					floodOut[a][b-1] = [0,0,0]
			if a>0 and b<(width-1) and visited[a-1][b+1]==0:
				if grayErode[a-1][b+1] == ref:
					queue.put([a-1,b+1])
					visited[a-1][b+1]=1
				else:
					floodOut[a-1][b+1] = [0,0,0]
			if b<(width-1) and visited[a][b+1]==0:
				if grayErode[a][b+1] == ref:
					queue.put([a,b+1])
					visited[a][b+1]=1
				else:
					floodOut[a][b+1] = [0,0,0]
			if a<(height-1) and b<(width-1) and visited[a+1][b+1]==0: 
				if grayErode[a+1][b+1] == ref:
					queue.put([a+1,b+1])
					visited[a+1][b+1]=1
				else:
					floodOut[a+1][b+1] = [0,0,0]
			if a<(height-1) and visited[a+1][b]==0:
				if grayErode[a+1][b] == ref:
					queue.put([a+1,b])
					visited[a+1][b]=1
				else:
					floodOut[a+1][b] = [0,0,0]
			if a<(height-1) and b>0 and visited[a+1][b-1]==0: 
				if grayErode[a+1][b-1] == ref:
					queue.put([a+1,b-1])
					visited[a+1][b-1]=1
				else:
					floodOut[a+1][b-1] = [0,0,0]

		return floodOut

	def cropImg(self,img):
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		height = img.shape[0]
		width = img.shape[1]

		check = 0

		Left = 0
		Right = 0
		Top = 0
		Bottom = 0

		for j in xrange(0, width):
			for i in xrange(0, height):
				if(img[i][j] != img[0][0]):
					Left = j
					check = 1
					break
			if(check == 1):
				break

		check = 0

		for j in xrange(width-1, -1, -1):
			for i in xrange(0, height):
				if(img[i][j] != img[0][0]):
					Right = j
					check = 1
					break
			if(check == 1):
				break

		check = 0

		for i in xrange(height-1, -1, -1):
			for j in xrange(0, width):
				if(img[i][j] != img[0][0]):
					Bottom = i
					check = 1
					break
			if(check == 1):
				break
		check = 0

		for i in xrange(0, height):
			for j in xrange(0, width):
				if(img[i][j] != img[0][0]):
					Top = i
					check = 1
					break
			if(check == 1):
				break

		crop_img = self.img[Top:Bottom, Left:Right]

		return crop_img

	def segImage(self,floodOut):
		grayCatImg = cv2.cvtColor(floodOut,cv2.COLOR_BGR2GRAY)
		i = grayCatImg.shape[0]/2
		for j in xrange(0, grayCatImg.shape[1]):
			if grayCatImg[i][j] != grayCatImg[0][0]:
				self.leftSegLine = j
				break

		for j in xrange(grayCatImg.shape[1]-1,-1,-1):
			if grayCatImg[i][j] != grayCatImg[0][0] :
				self.rightSegLine = j
				break 

		LPrev = self.leftSegLine
		RPrev = self.rightSegLine

		check = 0

		self.leftSegLine = 0
		self.rightSegLine = 0

		prevI = 0
		start = -1
		for i in xrange(grayCatImg.shape[0]/2, -1, -1):
			cnt=start
			for j in xrange(grayCatImg.shape[1]/2, -1, -1):
				if (cnt == 1) and (grayCatImg[i][j] != 0):
					cnt = 2
					start = 0
					prevI = i
					break
				if (cnt == start) and (grayCatImg[i][j] == 0):
					cnt = 1
			if (cnt == 1) and (start != -1):
				break


		for j in xrange(grayCatImg.shape[1]/2, -1, -1):
			if grayCatImg[prevI+1][j] == 0:
				prevJ1 = j
				break

		for j in xrange(grayCatImg.shape[1]/2, -1, -1):
			if grayCatImg[prevI][j] == 0 :
				prevJ2 = j
				break

		self.leftSegLine = min(prevJ1, prevJ2)

		prevI = 0
		start = -1
		for i in xrange(grayCatImg.shape[0]/2, -1, -1):
			cnt=start
			for j in xrange(grayCatImg.shape[1]/2, grayCatImg.shape[1]):
				if (cnt == 1) and (grayCatImg[i][j] != 0):
					cnt = 2
					start = 0
					prevI = i
					break
				if (cnt == start) and (grayCatImg[i][j] == 0):
					cnt = 1
			if(cnt == 1 and start != -1):
				break


		for j in xrange(grayCatImg.shape[1]/2, grayCatImg.shape[1]):
			if grayCatImg[prevI+1][j] == 0 :
				prevJ1 = j
				break

		for j in xrange(grayCatImg.shape[1]/2, grayCatImg.shape[1]):
			if grayCatImg[prevI][j] == 0:
				prevJ2 = j
				break

		self.rightSegLine = min(prevJ1, prevJ2)


		if(abs(grayCatImg.shape[1]/2-self.leftSegLine)<abs(LPrev-self.leftSegLine)):
			self.leftSegLine = LPrev
		if(abs(grayCatImg.shape[1]/2-self.rightSegLine)<abs(RPrev-self.rightSegLine)):
			self.rightSegLine = RPrev
		if(abs(LPrev/2-self.leftSegLine)<abs(LPrev-self.leftSegLine)):
			self.leftSegLine = LPrev
		if(abs(RPrev + (grayCatImg.shape[1]-RPrev)/2 - self.rightSegLine)<abs(RPrev-self.rightSegLine)):
			self.rightSegLine = RPrev

	def getSegLines(self):
		return self.leftSegLine,self.rightSegLine
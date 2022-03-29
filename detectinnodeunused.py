import cv2
import numpy as np
 
images = ["Cool.png","pog.png","pepeppo.png","HAHAUDIETHANOSSNAP.png"]
planets	= cv2.imread(images[3])
gray_img=cv2.cvtColor(planets,	cv2.COLOR_BGR2GRAY)
img	= cv2.medianBlur(gray_img,	1)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
 
#center
# cv2.imshow("HoughCirlces",	img)
# cv2.waitKey()
circles	= cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,param1=100,param2=30,minRadius=20,maxRadius=120)
circles	= np.uint16(np.around(circles))
 
for	i in circles[0,:]:
				#	draw	the	outer	circle
				cv2.circle(planets,(i[0],i[1]),i[2],(0,255,0),6)
				#	draw	the	center	of	the	circle
				cv2.circle(planets,(i[0],i[1]),2,(0,0,255),3)
 
cv2.imshow("HoughCirlces",	planets)
cv2.waitKey()
cv2.destroyAllWindows()
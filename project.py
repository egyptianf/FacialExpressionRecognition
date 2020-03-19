import cv2
import os
import glob
#Importing Image module from PIL package
#from PIL import Image
import matplotlib.pyplot as plt



im = cv2.imread('C:\\Users\\ahmedelsayed\\Desktop\\year3\\term2\\imageprocessing\\projectImage\\dataset\\anger\\S010_004_00000017.png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

plt.subplot(221)
plt.imshow(im,'gray')
plt.title('ORIGINAL')
plt.subplot(222)
plt.imshow(blur,'gray')
plt.title('GRAY')

#We will add code here
#We will also add code here



#My code is finished

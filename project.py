import cv2
import os
import platform
import glob
import numpy as np
import matplotlib.pyplot as plt
import math

operating_system = platform.system()
d = '/'
if operating_system == 'Windows':
    d = '\\'
img_dir = "projectImage" + d + "dataset"
data_path = os.path.join(img_dir, '*')

#Read the data set
files = glob.glob(data_path)
dataset = []

for f1 in files:
    dataset.append(os.path.basename(f1))
    dataset_path = os.path.join(f1,'*g')
    photos = glob.glob(dataset_path)
    dataset[-1] = []
    for f2 in photos:
        img = cv2.imread(f2)
        dataset[-1].append(img)
fig = plt.figure(figsize=(9, 9))
im = cv2.imread("projectImage" + d + "dataset" + d + "anger" + d + "S010_004_00000017.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = im

#Average filter to smooth images************************
avg_img = cv2.blur(gray.astype(np.float32),(3,3))

#Median filter to smooth images*************************
median_img = cv2.medianBlur(gray.astype(np.float32),3,3)

#Sobel filter to sharpen images*************************
imgSobX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) 
imgSobY = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3) 
imgSobel = imgSobX + imgSobY

#Perwitt filter to sharpen images***********************
kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_perx = cv2.filter2D(gray,-1,kernelx)
img_pery = cv2.filter2D(gray,-1,kernely)
imgper = img_perx + img_pery

'''Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. If ksize is set to [0 0], then ksize is computed from sigma values. 
sigmaXKernel(horizontal direction).
sigmaYKernel(vertical direction). If sigmaY=0, then sigmaX value is taken for sigmaY'''
blur = cv2.GaussianBlur(gray,(5,5),0)

#2D convolution to blur and sharpen image, it is a matrix(5x5) multiplied by 1/25
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(im,-1,kernel)

## Contrast Stretching to update range of image gray level
x1= np.min(im)
x2=np.max(im)

#suppose constant values of new dynamic range
y1 = 2
y2 = 7

modified_value=(math.ceil((y2-y1)/(x2-x1)))*(gray-x1)+y1

fig.add_subplot(3, 4, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.title("Image", fontdict={'fontsize': 15})

# Show the histogram of the image
histg = cv2.calcHist([im], [0], None, [256], [0, 256])

fig.add_subplot(3, 4, 2)
plt.plot(histg)
plt.title("Histogram", fontdict={'fontsize': 15})

# Perform Histogram Equalization 
equalized = cv2.equalizeHist(im)
fig.add_subplot(3, 4, 3)
plt.imshow(equalized, cmap=plt.cm.gray)
plt.title("Equalized", fontdict={'fontsize': 15})

equ_histg = cv2.calcHist([equalized], [0], None, [256], [0, 256])
fig.add_subplot(3, 4, 4)
plt.plot(equ_histg)
plt.title("Equalized Histogram", fontdict={'fontsize': 15})

fig.add_subplot(3, 4, 5)
plt.imshow(avg_img,'gray')
plt.title('average filter')

fig.add_subplot(3, 4, 6)
plt.imshow(median_img,'gray')
plt.title('median filter')

fig.add_subplot(3, 4, 7)
plt.imshow(imgSobel, cmap=plt.cm.gray)
plt.title('Sobel')

fig.add_subplot(3, 4, 8)
plt.imshow(imgper, cmap=plt.cm.gray)
plt.title('Perwitt')

fig.add_subplot(3, 4, 9)
plt.imshow(blur, cmap=plt.cm.gray)
plt.title('Gaussian')

fig.add_subplot(3, 4, 10)
plt.imshow(dst, cmap=plt.cm.gray)
plt.title('2D_convolution')

fig.add_subplot(3, 4, 11)
plt.imshow(modified_value,cmap=plt.cm.gray)
plt.title('contrast streatching')
plt.show()

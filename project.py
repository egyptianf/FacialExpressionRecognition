import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  

img_dir = "C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\projectImage\\dataset"
data_path = os.path.join(img_dir,'*')
files = glob.glob(data_path)
dataset = []

for f1 in files:
    dataset.append(os.path.basename(f1))
    dataset_path = os.path.join(f1,'*g')
    photos = glob.glob(dataset_path)
    dataset[-1] = []
    for f2 in photos:
        img = Image.open(f2)
        dataset[-1].append(img)
    print(dataset[-1])
im =Image.open( "C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\projectImage\\dataset\\anger\\S010_004_00000017.png")
plt.imshow(im)
#******************************************************

img = cv2.imread('C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\imageProject\\projectImage\\dataset\\anger\\S010_004_00000017.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#*******************************************************
avg_img = cv2.blur(gray.astype(np.float32),(3,3))
#*******************************************************
median_img = cv2.medianBlur(gray.astype(np.float32),3,3)
#*******************************************************

imgSobX = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) 
imgSobY = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3) 
imgSobel = imgSobX + imgSobY
#*******************************************************

kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_perx = cv2.filter2D(gray,-1,kernelx)
img_pery = cv2.filter2D(gray,-1,kernely)
imgper = img_perx + img_pery



plt.subplot(221)
plt.imshow(avg_img,'gray')
plt.title('average filter')
plt.plot()

plt.subplot(222)
plt.imshow(median_img,'gray')
plt.title('median filter')
plt.plot()

plt.subplot(223)
plt.imshow(imgSobel, cmap=plt.cm.gray)
plt.title('Sobel')
plt.plot()

plt.subplot(223)
plt.imshow(imgper, cmap=plt.cm.gray)
plt.title('Perwitt')
plt.plot()

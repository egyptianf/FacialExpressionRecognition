import cv2
import os
import glob
import numpy as np
import math


# Importing Image module from PIL package
#from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 9))
img_dir = "projectImage\\dataset"
data_path = os.path.join(img_dir, '*')
files = glob.glob(data_path)
dataset = []

for f1 in files:
    dataset.append(os.path.basename(f1))
    dataset_path = os.path.join(f1, '*g')
    photos = glob.glob(dataset_path)
    dataset[-1] = []
    for f2 in photos:
        img = cv2.imread(f2)
        dataset[-1].append(img)
    print(dataset[-1])
img= cv2.imread("C:\level 3\second simister\img processing\new_proj\FacialExpressionRecognition\projectImage\dataset\anger\\S010_004_00000017.png")
#2D_filter
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

fig.add_subplot(3, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.xticks([]), plt.yticks([])
fig.add_subplot(3, 3, 2),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
#contrast 

def getGrayColor(rgb):
    return rgb[0]


def setGrayColor(color):
    return [color, color, color]

plt.imshow(img)
ct = deepcopy(img)
r1 = 1000
s1 = 700
r2 = 500
s2 = 200

for i in range(len(img)):
    for j in range(len(img[i])):
        x = getGrayColor(img[i][j])
        if(0 <= x and x <= r1):
            ct[i][j] = setGrayColor(s1/r1 * x)
        elif(r1 < x and x <= r2):
            ct[i][j] = setGrayColor(((s2 - s1)/(r2 - r1)) * (x - r1) + s1)
        elif(r2 < x and x <= 255):
            ct[i][j] = setGrayColor(((255 - s2)/(255 - r2)) * (x - r2) + s2)

fig.add_subplot(3, 3, 3)
plt.imshow(img),plt.title('Original')
fig.add_subplot(3, 3, 4)
plt.imshow(ct),plt.title('contrast streatching')


plt.show()
#contrast in another way


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap=plt.cm.gray)
plt.title('Gray Level')


x1= np.min(img)
x2=np.max(img)

y1=(int)(input('enter y1:'))
y2=(int)(input('enter y2:'))


modified_value=(math.ceil((y2-y1)/(x2-x1)))*(gray-x1)+y1

fig.add_subplot(3, 3, 5)
plt.imshow(gray,'gray')
plt.title('GRAY')
plt.plot()

fig.add_subplot(3, 3, 6)
plt.imshow(modified_value,'gray')
plt.title('contrast')
plt.plot()



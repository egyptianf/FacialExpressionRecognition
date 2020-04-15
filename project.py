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
fig = plt.figure(figsize=(8, 8))
#fig1 = plt.figure(figsize=(6, 6))

im = cv2.imread("projectImage" + d + "dataset" + d + "anger" + d + "S022_005_00000030.png")

im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
height, width = im.shape
im = im[3:height-3, 5:width-5]
gray = im

def auto_canny(im, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(im)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(im, lower, upper)
	# return the edged image
	return edged

blur = cv2.GaussianBlur(im,(5,5),0)
wide = cv2.Canny(blur, 10, 200)
tight = cv2.Canny(blur, 225, 250)
auto = auto_canny(blur)
	# show the images
 

cv2.waitKey(0)
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

fig.add_subplot(4, 4, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.title("Image")


# Show the histogram of the image
histg = cv2.calcHist([im], [0], None, [256], [0, 256])


fig.add_subplot(4, 4, 2)
plt.plot(histg)
plt.title("Histogram")


# Perform Histogram Equalization 
equalized = cv2.equalizeHist(im)
equ_histg = cv2.calcHist([equalized], [0], None, [256], [0, 256])

fig.add_subplot(4, 4, 3)
plt.imshow(equalized, cmap=plt.cm.gray)
plt.title("Equalized")


blur = cv2.GaussianBlur(equalized,(5,5),0)
laplacian = cv2.Laplacian(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),cv2.CV_64F)
def Zero_crossing(im, laplacian):
    z_c_im = np.zeros(im.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, laplacian.shape[0] - 1):
        for j in range(1, laplacian.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [laplacian[i+1, j-1],laplacian[i+1, j],laplacian[i+1, j+1],laplacian[i, j-1],laplacian[i, j+1],laplacian[i-1, j-1],laplacian[i-1, j],laplacian[i-1, j+1]]
            d1 = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
 
 
            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
 
            if z_c:
                if laplacian[i,j]>0:
                    z_c_im[i, j] = laplacian[i,j] + np.abs(e)
                elif laplacian[i,j]<0:
                    z_c_im[i, j] = np.abs(laplacian[i,j]) + d1
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_im/z_c_im.max()*255
    z_c_im = np.uint8(z_c_norm)
 
    return z_c_im
edges = cv2.Canny(blur,100,200)

sobelx = cv2.Sobel((cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel((cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),cv2.CV_64F,0,1,ksize=5)  # y
sobel = sobelx + sobely

fig.add_subplot(4, 4, 4)
plt.plot(equ_histg)
plt.title("Equalized Histogram")

fig.add_subplot(4, 4, 5)
plt.imshow(avg_img,'gray')
plt.title('average filter')

fig.add_subplot(4, 4, 6)
plt.imshow(median_img,'gray')
plt.title('median filter')

fig.add_subplot(4, 4, 7)
plt.imshow(imgSobel, cmap=plt.cm.gray)
plt.title('Sobel')

fig.add_subplot(4, 4, 8)
plt.imshow(imgper, cmap=plt.cm.gray)
plt.title('Perwitt')

fig.add_subplot(4, 4, 9)
plt.imshow(blur, cmap=plt.cm.gray)
plt.title('Gaussian')

fig.add_subplot(4, 4, 10)
plt.imshow(dst, cmap=plt.cm.gray)
plt.title('2D_convolution')

fig.add_subplot(4, 4, 11)
plt.imshow(modified_value,cmap=plt.cm.gray)
plt.title('contrast streatching')

kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_perx = cv2.filter2D(blur,-1,kernelx)
img_pery = cv2.filter2D(blur,-1,kernely)
imgper = img_perx + img_pery


plt.subplot(4,4,12),plt.imshow( laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(4,4,13),plt.imshow(edges,cmap = 'gray')
plt.title('edge'), plt.xticks([]), plt.yticks([])
plt.subplot(4,4,14),plt.imshow(imgper,cmap = 'gray')
plt.title('Perwit'), plt.xticks([]), plt.yticks([])

plt.show()

'''
kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_perx = cv2.filter2D(blur,-1,kernelx)
img_pery = cv2.filter2D(blur,-1,kernely)
imgper = img_perx + img_pery

plt.subplot(3,3,1),plt.imshow(im,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(sobel,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(edges,cmap = 'gray')
plt.title('edge'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(imgper,cmap = 'gray')
plt.title('Perwit'), plt.xticks([]), plt.yticks([])

plt.show()

'''
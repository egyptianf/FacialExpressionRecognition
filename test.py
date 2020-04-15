import cv2
#from PIL import Image
from skimage.io import imread, imshow

import numpy as np
#from skimage.filters import gabor
#from skimage import io
from matplotlib import pyplot as plt  # doctest: +SKIP
from skimage.filters import prewitt_h,prewitt_v

#image = cv2.imread("C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\imageProject\\projectImage\\dataset\\happy\\S022_005_00000030.png")

#im = Image.open("C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\imageProject\\projectImage\\dataset\\happy\\S022_005_00000030.png")

image = imread("C:\\Users\\CRIZMA-MEGA STOR\\Desktop\\imageProject\\projectImage\\dataset\\happy\\S014_005_00000015.png", as_gray=True)
#imshow(image)
edges_prewitt_horizontal = prewitt_h(image)
#calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)

thresh = 255
im_bw = cv2.threshold(edges_prewitt_vertical, thresh, 255, cv2.THRESH_BINARY)[1]
imshow(edges_prewitt_vertical, cmap='gray')

image.shape
# imshow(image)
#pixel features
'''
features = np.reshape(image, (48*48))
print(features)
 # detecting edges in a coin image
 '''
filt_real, filt_imag = gabor(image, frequency=0.6)
plt.figure()            # doctest: +SKIP
io.imshow(filt_real)    # doctest: +SKIP
io.show()               # doctest: +SKIP
# less sensitivity to finer details with the lower frequency kernel
filt_real, filt_imag = gabor(image, frequency=0.1)
plt.figure()            # doctest: +SKIP
io.imshow(filt_real)    # doctest: +SKIP
io.show()               # doctest: +SKIP

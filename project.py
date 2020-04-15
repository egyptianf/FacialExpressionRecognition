import cv2
import os
import platform
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from image import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

operating_system = platform.system()
d = '/'
if operating_system == 'Windows':
    d = '\\'
img_dir = "projectImage" + d + "dataset"
data_path = os.path.join(img_dir, '*')
# Read the data set
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
fig = plt.figure(figsize=(8, 8))

images = []
image_path = "projectImage" + d + "dataset" + d + "anger" + d + "S022_005_00000030.png"
im = cv2.imread(image_path)
fearing_im = cv2.imread('projectImage' + d + 'dataset' + d + 'fear' + d + 'S011_003_00000013.png')
# im = fearing_im
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
height, width = im.shape
im = im[3:height - 3, 5:width - 5]
gray = im
images.append(Image('Original', im))

# Show the histogram of the image
histg = cv2.calcHist([im], [0], None, [256], [0, 256])
images.append(Image('Histogram', histg))
# Perform Histogram Equalization
equalized = cv2.equalizeHist(im)
equ_histg = cv2.calcHist([equalized], [0], None, [256], [0, 256])
images.append(Image('Equalized', equalized))
images.append(Image('Equalized Histogram', equ_histg))

blur = cv2.GaussianBlur(im, (5, 5), 0)
wide = cv2.Canny(blur, 10, 200)

cv2.waitKey(0)
# Average filter to smooth images************************
avg_img = cv2.blur(gray.astype(np.float32), (3, 3))
images.append(Image('Average Filter', avg_img))
# Median filter to smooth images*************************
median_img = cv2.medianBlur(gray.astype(np.float32), 3, 3)
images.append(Image('Median Filter', median_img))


def sobel(_gray):
    imgSobX = cv2.Sobel(_gray, cv2.CV_8U, 1, 0, ksize=3)
    imgSobY = cv2.Sobel(_gray, cv2.CV_8U, 0, 1, ksize=3)
    _imgSobel = imgSobX + imgSobY
    return _imgSobel


imgSobel = sobel(gray)
images.append(Image('Sobel', imgSobel))


def perwitt(_gray):
    kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    _img_perx = cv2.filter2D(_gray, -1, kernelx)
    _img_pery = cv2.filter2D(_gray, -1, kernely)
    _imgper = _img_perx + _img_pery
    return _imgper


imgper = perwitt(gray)
images.append(Image('Perwitt', imgper))

# 2D convolution to blur and sharpen image, it is a matrix(5x5) multiplied by 1/25
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(im, -1, kernel)
images.append(Image('2D Convolution', dst))


def contrast_stretch(_im):
    ## Contrast Stretching to update range of image gray level
    x1 = np.min(_im)
    x2 = np.max(_im)
    # suppose constant values of new dynamic range
    y1 = 2
    y2 = 7

    _modified_value = (math.ceil((y2 - y1) / (x2 - x1))) * (gray - x1) + y1
    return _modified_value


modified_value = contrast_stretch(im)
images.append(Image('Contrast Stretching', modified_value))
blur = cv2.GaussianBlur(equalized, (5, 5), 0)
images.append(Image('Gaussian', blur))
laplacian = cv2.Laplacian(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB), cv2.CV_64F)
images.append(Image('Laplacian', laplacian))


def Zero_crossing(laplacian):
    z_c_im = np.zeros(laplacian.shape)

    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood

    for i in range(1, laplacian.shape[0] - 1):
        for j in range(1, laplacian.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [laplacian[i + 1, j - 1], laplacian[i + 1, j], laplacian[i + 1, j + 1], laplacian[i, j - 1],
                         laplacian[i, j + 1], laplacian[i - 1, j - 1], laplacian[i - 1, j], laplacian[i - 1, j + 1]]
            d1 = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if laplacian[i, j] > 0:
                    z_c_im[i, j] = laplacian[i, j] + np.abs(e)
                elif laplacian[i, j] < 0:
                    z_c_im[i, j] = np.abs(laplacian[i, j]) + d1

    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_im / z_c_im.max() * 255
    z_c_im = np.uint8(z_c_norm)

    return z_c_im


edges = cv2.Canny(blur, 100, 200)
images.append(Image('Canny', edges))

sobelx = cv2.Sobel((cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)), cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel((cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)), cv2.CV_64F, 0, 1, ksize=5)  # y
sobel = sobelx + sobely
images.append(Image('Sobel', sobel))
inverted_laplacian = cv2.bitwise_not(laplacian)
smoothed_invLaplacian = cv2.medianBlur(inverted_laplacian.astype(np.float32), 3, 3)
images.append(Image('Smoothed', smoothed_invLaplacian))
smoothed_gray = cv2.cvtColor(smoothed_invLaplacian, cv2.COLOR_RGB2GRAY)
smoothed_gray = smoothed_gray.astype('uint8')


# Feature extraction section

sc = StandardScaler()
dataset = [im.flatten().astype('float32') for i in range(10000)]
dataset = sc.fit_transform(dataset)
pca = PCA(n_components=10)
print("data set shape", dataset.shape)
dataset = pca.fit_transform(dataset)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


def plot_images(_images, _fig, _plt):
    for i in range(len(_images)):
        _fig.add_subplot(4, 4, i + 1)
        if 'histogram' in _images[i].label.lower():
            _plt.plot(_images[i].image)
        else:
            _plt.imshow(_images[i].image, cmap=plt.cm.gray)
        _plt.title(_images[i].label)
    _plt.show()

plot_images(images, fig, plt)

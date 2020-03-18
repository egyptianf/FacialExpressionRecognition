import cv2
import os
import platform
import glob
import matplotlib.pyplot as plt

operating_system = platform.system()
d = '/'
if operating_system == 'Windows':
    d = '\\'
img_dir = "projectImage" + d + "dataset"
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
    # print(dataset[-1])


fig = plt.figure(figsize=(18, 18))
im = cv2.imread("projectImage" + d + "dataset" + d + "anger" + d + "S010_004_00000017.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
fig.add_subplot(2, 2, 1)
plt.imshow(im, cmap=plt.cm.gray)
plt.title("Image", fontdict={'fontsize': 30})
# Show the histogram of the image
histg = cv2.calcHist([im], [0], None, [256], [0, 256])
fig.add_subplot(2, 2, 2)
plt.plot(histg)
plt.title("Histogram", fontdict={'fontsize': 30})

# Perform Histogram Equalization
equalized = cv2.equalizeHist(im)
fig.add_subplot(2, 2, 3)
plt.imshow(equalized, cmap=plt.cm.gray)
plt.title("Equalized", fontdict={'fontsize': 30})

equ_histg = cv2.calcHist([equalized], [0], None, [256], [0, 256])
fig.add_subplot(2, 2, 4)
plt.plot(equ_histg)
plt.title("Equalized Histogram", fontdict={'fontsize': 30})

plt.show()

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


fig = plt.figure(figsize=(15, 15))
im = cv2.imread("projectImage" + d + "dataset" + d + "anger" + d + "S010_004_00000017.png")
fig.add_subplot(3, 3, 1)
plt.imshow(im)

plt.show()

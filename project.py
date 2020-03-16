import cv2
import os
import glob
# Importing Image module from PIL package
from PIL import Image
import matplotlib.pyplot as plt


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
        img = Image.open(f2)
        dataset[-1].append(img)
    print(dataset[-1])
im = Image.open("projectImage\\dataset\\anger\\S010_004_00000017.png")
plt.imshow(im)



#We will add code here
#We will also add code here



#My code is finished

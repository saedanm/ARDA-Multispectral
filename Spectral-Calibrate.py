
import cv2
import sys
import numpy as np
import time
from PIL import Image

from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector

def AR0134_rawTo8Bit(rawImage):
    newimage = cv2.normalize(rawImage, 0, 65535, cv2.NORM_MINMAX)
    #print(np.amin(newimage), np.amax(newimage))
    return cv2.cvtColor(np.uint8(newimage), cv2.COLOR_GRAY2RGB)


#Load save image (raw file)
#band_data_file = "shutter-test-image-20200727-0850/red-020-1.tif"
band_data_file = "image-test-20200727-1445/001-ir.tif"
band_data = np.asarray(Image.open(band_data_file))

#Flip image left to right
band_data = cv2.flip(band_data, 1)

#Convert to 8-bit image
image = AR0134_rawTo8Bit(band_data)

#Plot image
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(image, cmap="gray")
x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
pix = np.vstack((x.flatten(), y.flatten())).T

crop_data = []
crop_image = []
def mainselect(eclick, erelease):
    global crop_data
    global crop_image
    start = [int(eclick.xdata), int(eclick.ydata)]
    end = [int(erelease.xdata), int(erelease.ydata)]
    #Calculate average pixel value in the retanggle
    crop_image = image[start[1]:end[1], start[0]:end[0]]
    crop_data  = band_data[start[1]:end[1], start[0]:end[0]]
    print("Average crop area = %d"%np.average(crop_data))

lasso = RectangleSelector(ax1, mainselect)
plt.show()
plt.close()

print (crop_image.shape)


#Plot zoom image
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(crop_image)

x, y = np.meshgrid(np.arange(crop_image.shape[1]), np.arange(crop_image.shape[0]))
pix = np.vstack((x.flatten(), y.flatten())).T

def onselect(eclick, erelease):
    start = [int(eclick.xdata), int(eclick.ydata)]
    end = [int(erelease.xdata), int(erelease.ydata)]
    #Calculate average pixel value in the retanggle
    selection = crop_data[start[1]:end[1], start[0]:end[0]]
    
    print("Average pixel value = %d"%np.average(selection))

lasso = RectangleSelector(ax1, onselect)
plt.show()


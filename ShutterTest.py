
import cv2
import sys
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt
import matplotlib.colors

from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector

import time

#--------------------------------------------------------------------------------------------------------------
#SpectralIntensity is the spectral value converted to image intensity
#Offset is the dark current intensity of each pixel
def ARDA_Reflectance(rawImage, SpectralIntensity, Offset):

    reflectedImage = np.zeros((964,1280), dtype=np.float32)
    
    #Limit pixel intensity to within dark_current and maximum intensity
    reflectedImage = np.clip(rawImage, a_min = Offset, a_max = SpectralIntensity)
    
    #Convert image intensity to reflectance
    #1) Remove dark current offset
    reflectedImage = reflectedImage - Offset
    
    #2) Calculate reflectance ratio
    reflectedImage = reflectedImage/SpectralIntensity
    
    #Limit reflectance ratio to 1
    #reflectedImage = np.clip(reflectedImage, a_min = 0.0, a_max = 1.0)

    return reflectedImage
    
#-----------------------------------------------------------------------------------
def AR0134_rawTo8Bit(rawImage):
    newimage = cv2.normalize(rawImage, 0, 65535, cv2.NORM_MINMAX)
    #print(np.amin(newimage), np.amax(newimage))
    return cv2.cvtColor(np.uint8(newimage), cv2.COLOR_GRAY2RGB)

#------------------------------------ MAIN PROGRAM ---------------------------------
#folder_name = "shutter-test-image-20200727-0850/"
#folder_name = "shutter-test-image-20200727-0915/"
#folder_name = "shutter-test-image-20200727-1200/"
folder_name = "shutter-test-image-20200727-1210-E/"
#folder_name = "shutter-test-image-20200727-1320/"
folder_name = "shutter-test-image-20200727-1330-E/"

bins = np.linspace(0, 4096, num=32)

bins_str=[]
for i in range(1, len(bins)):
    bins_str.append(str("%d"%bins[i]))
    


total_red_spec = 0
total_ir_spec = 0

for i in range(100, 0, -10):

    shutter_str = str(i).zfill(3)

    red_data_file = folder_name + "red-" + shutter_str + "-1.tif"
    red_spectral_file = folder_name + "red-" + shutter_str + "-1.txt"
    
    ir_data_file = folder_name + "ir-" + shutter_str + "-1.tif"
    ir_spectral_file = folder_name + "ir-" + shutter_str + "-1.txt"
    
    #Read red spectral value
    sp = open(red_spectral_file, "r")
    data = sp.readline()
    data = data.strip()
    data = data.split("\t")
    red_spectral = float(data[0])
    sp.close()

    #Read red spectral value
    sp = open(ir_spectral_file, "r")
    data = sp.readline()
    data = data.strip()
    data = data.split("\t")
    ir_spectral = float(data[1])
    sp.close()
    
    total_red_spec = total_red_spec + red_spectral
    total_ir_spec = total_ir_spec + ir_spectral

    #Load save image (raw file)
    red_data = np.asarray(Image.open(red_data_file))
    ir_data = np.asarray(Image.open(ir_data_file))
    
    #Process red image
    total_red_points_4000 = np.sum((red_data>=4000).astype(int))
    print("red points>4000 --> %d"%(total_red_points_4000))
    
    red_chart_name = "Graph-Distribution-VS-Shutter/" + "red-" + shutter_str + ".png"
    [red_hist, red_bin]  = np.histogram(red_data, bins=bins, density=False)
    graph = plt.bar(bins_str, red_hist, color='red')
    plt.ylabel("Nunber of pixels")
    plt.xlabel("Pixel value")
    plt.xticks(rotation=90)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig(red_chart_name, dpi=150)
    plt.close()


    #Process ir image
    total_ir_points_4000 = np.sum((ir_data>=4000).astype(int))
    print("IR points>4000 --> %d"%(total_ir_points_4000))
    
    ir_chart_name = "Graph-Distribution-VS-Shutter/" + "ir-" + shutter_str + ".png"
    [ir_hist, ir_bin]  = np.histogram(ir_data, bins=bins, density=False)
    graph = plt.bar(bins_str, ir_hist, color='brown')
    plt.ylabel("Nunber of pixels")
    plt.xlabel("Pixel value")
    plt.xticks(rotation=90)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig(ir_chart_name, dpi=150)
    plt.close()


print("Red %0.1f\tIR %0.1f\n"%(total_red_spec/10, total_ir_spec/10))

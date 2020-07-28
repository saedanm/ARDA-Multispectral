
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
#Find homography transformation from query image to template image
def ORB_Feature_Homography(imageQuery, imageTemplate):
    #Align img1 to img2
    img1 = imageQuery
    img2 = imageTemplate
    
    orb_detector = cv2.ORB_create(5000)
      
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
      
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
      
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
      
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
      
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)
      
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    #print(len(matches))

    #Get matched pair-points
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
      
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    return homography
    
#--------------------------------------------------------------------------------------------------------------
#SpectralIntensity is the spectral value converted to image intensity
#Offset is the dark current intensity of each pixel
def ARDA_Reflectance(rawImage, SpectralIntensity, Offset):

    reflectedImage = np.zeros((964,1280), dtype=np.float32)
    
    #Limit pixel intensity to within dark_current and maximum intensity
    reflectedImage = np.clip(rawImage, a_min=Offset, a_max=SpectralIntensity)
    
    #Convert image intensity to reflectance
    #1) Remove dark current offset
    reflectedImage = reflectedImage - Offset
    
    #2) Calculate reflectance ratio
    reflectedImage = reflectedImage/SpectralIntensity
    
    #Limit reflectance ratio to 1
    #reflectedImage = np.clip(reflectedImage, a_min = 0.0, a_max = 1.0)

    return reflectedImage

#--------------------------------------------------------------------------------------------------------------
def GetSpectralData(imagePIL):
    return [imagePIL.tag[37000], imagePIL.tag[37001], imagePIL.tag[37002], imagePIL.tag[37003]]

#------------------------------------ MAIN PROGRAM ---------------------------------
folder_name = "image-test-20200727-1445/"
folder_name = "field-test-20200727-1545/"

for i in range(1,20):

    index = str(i).zfill(3)

    red_data_file = folder_name + index +"-red.tif"
    ir_data_file = folder_name + index +"-ir.tif"
    spectral_data_file = folder_name + index+"-spectral.txt"

    #Read spectral value from spectral data file
    sp = open(spectral_data_file, "r")
    data = sp.readline()
    data = data.strip()
    data = data.split("\t")

    red_spectral = float(data[0])
    ir_spectral = float(data[1])

    #Load save image (raw file)
    red_data = np.asarray(Image.open(red_data_file))
    ir_data = np.asarray(Image.open(ir_data_file))

    #Calculate spectral intensity and offset
    #Based on value obtain 27 July 2020
    red_offset= 502.48 +0.0338*red_spectral
    ir_offset = 605.07 -0.0067*ir_spectral

    red_spectral_intensity = 0.8769*red_spectral + 797.85
    ir_spectral_intensity = 0.798*ir_spectral - 534.52

    red_reflectance = ARDA_Reflectance(red_data, red_spectral_intensity, red_offset)
    ir_reflectance = ARDA_Reflectance(ir_data, ir_spectral_intensity, ir_offset)

    print("%d-->Red spectral %0.2f : IR spectral %0.2f : Avg Red = %d : Avg IR = %d"%(i,red_spectral, ir_spectral,np.mean(red_data), np.mean(ir_data)))

    #No fliping for newer data from 27 July
    #Flip image horizontally to make it look normal
    #red_reflectance = cv2.flip(red_reflectance, 1)
    #ir_reflectance = cv2.flip(ir_reflectance, 1)

    start_time = time.time()

    #Determine homography matrix using ORB feature point matching
    red_grey = np.uint8(red_reflectance*255)
    ir_grey = np.uint8(ir_reflectance*255)
    Homography = ORB_Feature_Homography(ir_grey, red_grey)

    #Align IR to red image
    ir_reflectance= cv2.warpPerspective(ir_reflectance, Homography, (1280, 964))
    end_time = time.time()

    #print("Time taken %0.3f"%(end_time-start_time))
    

    display = np.hstack((ir_reflectance,red_reflectance))
    figure = plt.imshow(display, cmap="gray")
    figure.set_clim(0.0,1.0)
    plt.colorbar()
    plt.show()
    
    '''
    #Crop image
    [height, width]         = ir_reflectance.shape
    crop_ir_reflectance     = ir_reflectance#ir_reflectance[80:height-1, 90:width-1]
    crop_red_reflectance    = red_reflectance#red_reflectance[80:height-1, 90:width-1]

    #Calculate NDVI
    np.seterr(divide='ignore')
    ndvi_image = np.divide(crop_ir_reflectance - crop_red_reflectance, crop_ir_reflectance + crop_red_reflectance)
    #Just in case we got divided by ZERO, replacing the NaN value with 0
    ndvi_image[np.isnan(ndvi_image)]=0

    #Plot image
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("user", ["black","white","green"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("user", ["red", "yellow" ,"green"])
    

    plt.imshow(ndvi_image, cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.show()
    '''



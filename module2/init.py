#Includes
import cv2
import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import *
from skimage.filters import *
import os
from find_circles import *
from get_questions import *

#Reading Images
def appendImage(imgPath,imList):
    #Image is appended in GrayScale
    imList.append(rgb2gray(cv2.imread(imgPath,0)))
    return
#Geting the Choices
import cv2
from skimage.filters import sobel
from skimage.morphology import binary_erosion,binary_dilation
import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon

def Closing(img,lv=1):
    if lv == 0:
        return img
    return binary_erosion(Closing(binary_dilation(img),lv-1))

def Draw_Contours(img):
    contours = find_contours(img,0.8)
    cont_stp = np.zeros(img.shape)
    X_dim=[]                 #(xmin,xmax)
    Y_dim=[]                 #(ymin,ymax)
    for c in contours:
        Xmax,Xmin,Ymax,Ymin = max(c[:,1]), min(c[:,1]), max(c[:,0]), min(c[:,0])
        rr,cc = polygon([Ymin, Ymax, Ymax, Ymin],
                            [Xmin, Xmin, Xmax, Xmax], shape=img.shape)
        cont_stp[rr,cc]=1
    return cont_stp

def Distribute_Area(img,Area_Ratio=None):
    cont_stp = np.zeros(img.shape)
    contours = find_contours(img,0.8)
    Area = []
    X_dim=[]                 #(xmin,xmax)
    Y_dim=[]                 #(ymin,ymax)
    for c in contours:
        Xmax,Xmin,Ymax,Ymin = max(c[:,1]), min(c[:,1]), max(c[:,0]), min(c[:,0])
        X_dim.append([Xmin,Xmax])
        Y_dim.append([Ymin,Ymax])
        Area.append((Xmax-Xmin)*(Ymax-Ymin))     
    vari_Area = np.sqrt(np.var(np.array(Area)))
    men_Area = np.mean(np.array(Area))
    if(Area_Ratio is not None):
        vari_Area = Area_Ratio*men_Area
    for c in range (0, len(contours)):
        if abs(Area[c]-men_Area) <= vari_Area:
            rr,cc = polygon([Y_dim[c][0], Y_dim[c][1], Y_dim[c][1], Y_dim[c][0]],
                            [X_dim[c][0], X_dim[c][0], X_dim[c][1], X_dim[c][1]], shape=img.shape)
            cont_stp[rr,cc]=1
    return cont_stp

def getChoices(img):
    img = cv2.GaussianBlur(img,(11,11),2,2)
    binary = sobel(img)    
    binary = np.round(binary*255).astype('uint8')
    binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    Closed = np.round(Closing(binary) * 255).astype('uint8')
    Closed = binary_erosion(Closed)
    
    cont_stp1 = Draw_Contours(Closed)          #To merge Contours
    cont_stp2 = Distribute_Area(cont_stp1)     #To Distribute according to Area Uniform Distribution
    cont_stp3 = Distribute_Area(cont_stp2)     #To Distribute according to Area Uniform Distribution
    cont_stp4 = Distribute_Area(cont_stp3,0.3) #To Get according to a little shift to the mean in Area Uniform Distribution
    Final = Draw_Contours(cont_stp4)           #Get the Final Contours Level
    contours = find_contours(Final,0.8)
    Ch = np.zeros((len(contours),4))
    i=0
    for c in contours:
        Xmax,Xmin,Ymax,Ymin = max(c[:,1]), min(c[:,1]), max(c[:,0]), min(c[:,0])
        Ch[i]=[Xmin,Xmax,Ymin,Ymax]
        i+=1
    return np.round(Ch).astype('uint32')

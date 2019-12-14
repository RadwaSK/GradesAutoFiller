#Geting the Circles

def Closing(img,lv=1):
    if lv == 0:
        return img
    return binary_erosion(Closing(binary_dilation(img),lv-1))


def getCircles(img):
    img = cv2.GaussianBlur(img,(11,11),2,2)
    binary = sobel(img)
    binary = np.round(binary*255).astype('uint8')
    binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    Closed = np.round(Closing(binary) * 255).astype('uint8')
    ret = cv2.HoughCircles(Closed,cv2.HOUGH_GRADIENT,1.7,minDist = 15,param1=50,param2=30,minRadius=20,maxRadius=24)
    
    if ret is not None:
        ret = np.round(ret).astype('uint16')
        #Get the peak not the median [IMPORTANT]
        radi = np.median(ret[0,:,2])
        #New Radius is 90% of the peak radius
        radi = (radi*0.9).astype('uint16')
        ret[0,:,2] = radi
		return ret[0] 
    return None

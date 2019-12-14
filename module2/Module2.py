from init import *

DataSet = os.listdir('dataset')
imagesList = []
for imgName in DataSet:
    appendImage('dataset/' + imgName ,imagesList)


Circles = getCircles(imagesList[3])
Rows = getRows(getCols())
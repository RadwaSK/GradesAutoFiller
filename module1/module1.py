import numpy as np
import cv2 as cv
import math
import os
import pytesseract
from scipy.misc.pilutil import imresize
import cv2 #version 3.2.0
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from xlwt import Workbook

#################################################################################

# classifier for hand-written images

DIGIT_WIDTH = 10
DIGIT_HEIGHT = 20
IMG_HEIGHT = 28
IMG_WIDTH = 28
CLASS_N = 10  # 0-9
# This method splits the input training image into small cells (of a single digit) and uses these cells as training data.
# The default training image (MNIST) is a 1000x1000 size image and each digit is of size 10x20. so we divide 1000/10 horizontally and 1000/20 vertically.
def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells
def load_digits(fn):
    print('loading "%s for training" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (DIGIT_WIDTH, DIGIT_HEIGHT))
    resized_digits = []
    for digit in digits:
        resized_digits.append(imresize(digit, (IMG_WIDTH, IMG_HEIGHT)))
    labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
    return np.array(resized_digits), labels
def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img,
                 orientations=10,
                 pixels_per_cell=(5, 5),
                 cells_per_block=(1, 1),
                 visualise=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)
# define a custom model in a similar class wrapper with train and predict methods
class KNN_MODEL():
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()
class SVM_MODEL():
    def __init__(self, num_feats, C=1, gamma=0.1):
        self.model = cv2.ml.SVM_create()
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_RBF)  # SVM_LINEAR, SVM_RBF
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.features = num_feats

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        results = self.model.predict(samples.reshape(-1, self.features))
        return results[1].ravel()
def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]
    final_bounding_rectangles = []
    # find the most common heirarchy level - that is where our digits's bounding boxes are
    u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]

    for r, hr in zip(bounding_rectangles, hierarchy):
        x, y, w, h = r
        # this could vary depending on the image you are trying to predict
        # we are trying to extract ONLY the rectangles with images in it (this is a very simple way to do it)
        # we use heirarchy to extract only the boxes that are in the same global level - to avoid digits inside other digits
        # ex: there could be a bounding box inside every 6,9,8 because of the loops in the number's appearence - we don't want that.
        # read more about it here: https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
        if ((w * h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            final_bounding_rectangles.append(r)

    return final_bounding_rectangles
def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)
    blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    blank_image.fill(255)
    numbers = []

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    digits_rectangles = get_digits(contours, hierarchy)  # rectangles of bounding the digits in user image

    for rect in digits_rectangles:
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_digit = imgray[y:y + h, x:x + w]
        im_digit = (255 - im_digit)
        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))

        hog_img_data = pixels_to_hog_20([im_digit])
        pred = model.predict(hog_img_data)
        cv2.putText(im, str(int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        numbers.append(str(int(pred[0])))
        cv2.putText(blank_image, str(int(pred[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

    # plt.imshow(im)
    cv2.imwrite("original_overlay.png", im)
    cv2.imwrite("final_digits.png", blank_image)
    cv2.destroyAllWindows()
    return numbers
def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]  # row-wise ordering
# this function processes a custom training image
# see example : custom_train.digits.jpg
# if you want to use your own, it should be in a similar format
def load_digits_custom(img_file, ):
    train_data = []
    # pd.read_csv('train.csv')
    # train_data=
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours, hierarchy)  # rectangles of bounding the digits in user image

    # sort rectangles accoring to x,y pos so that we can label them
    digits_rectangles.sort(key=lambda x: get_contour_precedence(x, im.shape[1]))

    for index, rect in enumerate(digits_rectangles):
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im_digit = imgray[y:y + h, x:x + w]
        im_digit = (255 - im_digit)

        im_digit = imresize(im_digit, (IMG_WIDTH, IMG_HEIGHT))
        train_data.append(im_digit)
        train_target.append(start_class % 10)

        if index > 0 and (index + 1) % 10 == 0:
            start_class += 1
    cv2.imwrite("training_box_overlay.png", im)

    return np.array(train_data), np.array(train_target)
# ------------------data preparation--------------------------------------------
def Num_Classifier(img_test):
    TRAIN_MNIST_IMG = 'digits.png'
    TRAIN_USER_IMG = 'custom_train_digits.jpg'
    TEST_USER_IMG = img_test
    # digits, labels = load_digits(TRAIN_MNIST_IMG) #original MNIST data (not good detection)
    digits, labels = load_digits_custom(
        TRAIN_USER_IMG)  # my handwritten dataset (better than MNIST on my handwritten digits)

    print('train data shape', digits.shape)
    print('test data shape', labels.shape)

    digits, labels = shuffle(digits, labels, random_state=256)
    train_digits_data = pixels_to_hog_20(digits)
    X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.33, random_state=42)

    # ------------------training and testing----------------------------------------

    model = KNN_MODEL(k=7)
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, preds))

    model = KNN_MODEL(k=7)
    model.train(train_digits_data, labels)
    numbers = proc_user_img(TEST_USER_IMG, model)

    model = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, preds))

    model = SVM_MODEL(num_feats=train_digits_data.shape[1])
    model.train(train_digits_data, labels)
    proc_user_img(TEST_USER_IMG, model)
    w = ""
    for i in reversed(numbers):
        w += i + ""

    return w
##################################################################################
def NumDetect_OCR(img):
    output=cv.imread(img)
    #RGB 2 GRAY
    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    #cv.imshow("imageResized GS",output)
    result=np.copy(output)
    ret, thresh = cv.threshold(result, 100, 255, 0)
    # cv.imshow("threshold result",thresh)
    #resize image
    scale_percentw = 250
    scale_percenth=300
    width = int(thresh.shape[1] * scale_percentw / 100)
    height = int(thresh.shape[0] * scale_percenth / 100)
    dsize = (width, height)
    output = cv.resize(thresh, dsize)
    #output=thresh2
    #cv.imwrite("white92.png",output
    SE = np.ones((3, 3), np.uint8)
    output=cv.erode(output,SE)
    cv.imwrite(".png",output)
    #OCR
    text = pytesseract.image_to_string(output, lang="eng")  #Specify language to look after!
    cv.imwrite("secnum%s.png" % text, output)
    return text

def NameDetect_OCR(img):
    output=cv.imread(img)
    #RGB 2 GRAY
    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    #cv.imshow("imageResized GS",output)
    #invertthresh
    ret, thresh = cv.threshold(output,150, 255, 0)
    #cv.imshow("threshold",thresh)
    result=np.copy(output)
    # Remove horizontal lines
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40,1))
    remove_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(remove_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(result, [c], -1, (255,255,255), 2)
    # cv.imshow('H',result)
    ret, thresh2 = cv.threshold(result, 150, 255, 0)
    #cv.imshow("threshold result",thresh2)
    #resize image
    scale_percentw = 250
    scale_percenth=300
    width = int(thresh2.shape[1] * scale_percentw / 100)
    height = int(thresh2.shape[0] * scale_percenth / 100)
    dsize = (width, height)
    output = cv.resize(thresh2, dsize)
    #cv.imshow("big",output)
    #OCR
    text = pytesseract.image_to_string(output, lang="ara")  #Specify language to look after!
    return text

def ReadImages(directory):
    fnames = os.listdir(directory)
    to_return = []
    for fn in fnames:
        path = os.path.join(directory, fn)
        gray_scale_image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY)
        to_return.append((fn, gray_scale_image))

    return to_return

def ResizeImage(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def RemoveDuplicates(lines):
    for i, (rho, theta) in enumerate(lines):
        for j, (rho2, theta2) in enumerate(lines):
            if j == i:
                continue
            deltaRho = abs(abs(rho) - abs(rho2))
            deltaTheta = abs(abs(theta) - abs(theta2))
            if deltaRho < 15 and deltaTheta < 25.0/180.0:
                del lines[j]
    return lines

def SortLinesList(lines):
    vertical = []
    horizontal = []
    for rho, theta in lines:
        if theta > 1 and theta < 2:
            horizontal.append((rho, theta))
        else:
            vertical.append((rho, theta))
    return horizontal, vertical

def FixList(list):
    newList = []
    for item in list:
        newList.append(item[0])
    return newList

def Closing(img):
    SE = np.ones((5, 5), np.uint8)
    SE2 = np.ones((3, 3), np.uint8)
    ret, img = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
    img = cv.dilate(img, SE)
    img = cv.erode(img, SE)
    img = cv.dilate(img, SE)
    return img

def AddLines(lines, img):
    if lines is not None:
        for rho, theta in lines:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #print("line at (x,y) = (", x0, ", ", y0, ") has rho, theta = ", rho, theta)
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    return lines, img

def CheckCorrectOrDash(orig_img):
    h, w = orig_img.shape
    #threshold the image
    ret, img = cv.threshold(orig_img, 150, 255, cv.THRESH_BINARY)

    #divide it into two parts
    left, right = img[:, 0:int(w / 2)], img[:, int(w / 2):w]
    
    #get canny edges in both halves
    canny_left_img = cv.Canny(left, threshold1=50, threshold2=200)
    if (canny_left_img is None):
        return ''

    canny_right_img = cv.Canny(right, threshold1=50, threshold2=200)
    if (canny_right_img is None):
        return ''

    #get Hough lines in both parts
    lines_left_temp = cv.HoughLinesP(canny_left_img, 1, np.pi / 25.0, 5)
    if (lines_left_temp is None):
        return ''
    lines_left = FixList(lines_left_temp)
    
    lines_right_temp = cv.HoughLinesP(canny_right_img, 1, np.pi / 25.0, 5)
    if (lines_right_temp is None):
        return ''
    lines_right = FixList(lines_right_temp)

    # get max line in length in left image and right image
    max_left_line = []
    max_right_line = []
    max_len_left = 0
    max_len_right = 0

    for x1, y1, x2, y2 in lines_left:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_len_left:
            max_len_left = length
            max_left_line = [x1, y1, x2, y2]
        
    for x1, y1, x2, y2 in lines_right:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_len_right:
            max_len_right = length
            max_right_line = [x1, y1, x2, y2]
        
    # check angle between two lines
    x1, y1, x2, y2 = max_left_line[0], max_left_line[1], max_left_line[2], max_left_line[3]
    ang1 = math.atan(float(y2 - y1) / (x2 - x1))
    
    x1, y1, x2, y2 = max_right_line[0], max_right_line[1], max_right_line[2], max_right_line[3]
    ang2 = math.atan(float(y2 - y1) / (x2 - x1))
    
    if abs(ang1) < 0.2 and abs(ang2) < 0.2:
        # cell is dash
        return 'dash'
    else:
        # cell is correct
        return 'correct'

def ExtractCells(pic):
    img = pic.copy()
    imgCopy = pic.copy()

    # get canny edge image
    canny_img = cv.Canny(img, 50, 350)

    # get hough lines
    linesTemp = cv.HoughLines(canny_img, 1, np.pi / 180.0, 270)

    # just changing data structure
    lines = FixList(linesTemp)

    # removing duplicate hough lines
    RemoveDuplicates(lines)

    # separating lines into vertical and horizontal
    hLines, vLines = SortLinesList(lines)

    # add hough lines to the original image
    lines, img = AddLines(lines, img)

    #close the image to get an image with hough lines only and white background
    closed_binary_image = Closing(img)

    # find contours in the image
    _, contours, hierachy = cv.findContours(closed_binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Trying to get y coordinates of the last row
    contours = np.delete(contours, np.s_[0:5])
    avg_height = 27
    contours_info = []
    w_min = 30
    w_max = 200

    # store x, y, w, h of right contours
    for cont in contours:
        x, y, w, h = cv.boundingRect(cont)

        if (w < w_min or w > w_max + 20):
            continue

        if (h > avg_height+7 or h < avg_height - 7):
            continue

        contours_info.append([y, x, w, h])

    # sort according to y value of the contours, to have the last row first, up to first row
    np.sort(contours_info)

    # just for testing purpose
    # imgCopy = cv.drawContours(imgCopy, contours, -1, (255, 255, 255), 2)
    # cv.imshow("contour image", ResizeImage(imgCopy, height=775))

    # can be changed according to the number of students in the sheet
    folder_num = 34
    last_width = 0
    cnt = 0

    # loop on contours, cut image, export it into seperate folders
    for contour in contours_info:
        y, x, w, h = contour
        cv.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Crop the result
        final_image = imgCopy[y:y + h + 1, x:x + w + 1]

        # check if I filled all student folders already, then the rest are false contours
        if (folder_num <= 0):
            break

        # make directory imgs/student*folder_num*
        dir = 'imgs/student%s' % folder_num

        if not os.path.exists(dir):
            os.makedirs(dir)

        # file name
        fn = '%s/cell%s.png' % (dir, cnt)
        # count of cells
        cnt += 1
        cv.imwrite(fn, final_image)
        if last_width > 150:
            folder_num -= 1
            cnt = 0
        last_width = w

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# __main__
dataset = ReadImages(r'dataset_module1')
ExtractCells(dataset[1][1])
folder_num = 34
# Workbook is created
wb = Workbook()
# add_sheet is used to create sheet.
excel = wb.add_sheet('el dragat', cell_overwrite_ok=True)
print('folders numbers', folder_num)
for i in range(0,folder_num):
    file_nm = "imgs/student%s" % (i+1)
    file_length=len([f for f in os.listdir(file_nm)])
    print('file length :', file_length)
    for j in range(0, file_length):
        file_name= file_nm + "/cell%s.png" % (j)
        print(file_name)
        image = cv.imread(file_name)
        if (image is None):
            break
        shape=np.shape(image)
        height=shape[0]
        width=shape[1]

        # height,width=image.shape

        if (width > 150): # NAME
            Name_text = NameDetect_OCR(file_name)
            excel.write(i, 2, Name_text)

        elif(width < 70 and width > 55 ): # NUMBER COMPUTER
            Num_text=NumDetect_OCR(file_name)
            excel.write(i, 0, Num_text)

        else:  # NUMBER/correct/dash HANDWRITTEN WRITTEN

            Num_written=Num_Classifier(file_name)
            if(Num_written == ""):
                string = CheckCorrectOrDash(image)
                if (string == 'dash'):
                	Num_written = '0'
                elif (string == 'correct':
                	Num_written = '5'
                else:
                    break
            excel.write(i, max(j, 3), Num_written)


wb.save('el drgaaat.xls')

cv.waitKey()

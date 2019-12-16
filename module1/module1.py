import numpy as np
import cv2 as cv
import math
import os


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

def HoughLinesManual(pic):
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

    closed_binary_image = Closing(img)

    # find contours in the image
    contours, hierachy = cv.findContours(closed_binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Trying to get y coordinates of the last row
    contours = np.delete(contours, np.s_[0:5])
    avg_height = 27
    contours_info = []
    y_max = -1
    w_min = 30
    w_max = 200

    # store x, y, w, h of contours
    for cont in contours:
        x, y, w, h = cv.boundingRect(cont)

        if (w < w_min or w > w_max + 20):
            continue
        if (h > avg_height+7 or h < avg_height - 7):
            continue

        contours_info.append([y, x, w, h])

        if y > y_max:
            y_max = y

    np.sort(contours_info)

    imgCopy = cv.drawContours(imgCopy, contours, -1, (255, 255, 255), 2)
    cv.imshow("contour image", ResizeImage(imgCopy, height=775))
    temp_list = []

    folder_num = 34
    last_width = 0
    cnt = 0

    for i, contour in enumerate(contours_info):
        y, x, w, h = contour
        cv.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # Crop the result
        final_image = imgCopy[y:y + h + 1, x:x + w + 1]
        # check if name
        if (folder_num <= 0):
            break
        # make directory imgs
        dir = 'imgs/student%s' % folder_num
        if not os.path.exists(dir):
            os.makedirs(dir)
        fn = '%s/cell%s.png' % (dir, cnt)
        cnt += 1
        cv.imwrite(fn, final_image)
        if last_width > 150:
            folder_num -= 1
            cnt = 0
        last_width = w

# __main__
dataset = ReadImages(r'dataset_module1')
HoughLinesManual(dataset[1][1])
cv.waitKey()
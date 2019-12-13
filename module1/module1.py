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

# def hough_transform_p(image, template, tableCnt):
#     # go through each horizontal line (aka row)
#     rows = []
#     for i, h in enumerate(horizontal):
#         if i < len(horizontal)-1:
#             row = []
#             for j, v in enumerate(vertical):
#                 if i < len(horizontal)-1 and j < len(vertical)-1:
#                     # every cell before last cell
#                     # get width & height
#                     width = horizontal[i+1][1] - h[1]
#                     height = vertical[j+1][0] - v[0]
#
#                 else:
#                     # last cell, width = cell start to end of image
#                     # get width & height
#                     width = tW
#                     height = tH
#                 tW = width
#                 tH = height
#
#                 # get roi (region of interest) to find an x
#                 roi = img[h[1]:h[1]+width, v[0]:v[0]+height]
#
#                 # save image (for testing)
#                 dir = 'imgs/table%s' % (tableCnt+1)
#                 if not os.path.exists(dir):
#                     os.makedirs(dir)
#                 fn = '%s/roi_r%s-c%s.png' % (dir, i, j)
#                 cv.imwrite(fn, roi)
#
#                 # if roi contains an x, add x to array, else add _
#                 roi_gry = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
#                 ret, thresh = cv.threshold(roi_gry, 127, 255, 0)
#                 contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#                 if len(contours) > 1:
#                     # there is an x for 2 or more contours
#                     row.append('x')
#                 else:
#                     # there is no x when len(contours) is <= 1
#                     row.append('_')
#             row.pop()
#             rows.append(row)
#
#     # save image (for testing)
#     fn = os.path.splitext(image)[0] + '-hough_p.png'
#     cv.imwrite('imgs/'+fn, img)
#
#
# def process(images):
#     for i, img in enumerate(images):
#         # perform probabilistic hough transform on each image
#         hough_transform_p(img, templates[0], i)

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

    # seperating lines into vertical and horizontal
    hLines, vLines = SortLinesList(lines)

    # add hough lines to the original image
    lines, img = AddLines(lines, img)


    # rows = []
    # for i, h in enumerate(hLines):
    #     if i < len(hLines)-1:
    #         row = []
    #         for j, v in enumerate(vLines):
    #             if i < len(hLines)-1 and j < len(vLines)-1:
    #                 # every cell before last cell
    #                 # get width & height
    #                 width = hLines[i+1][1] - h[1]
    #                 height = vLines[j+1][0] - v[0]
    #
    #             else:
    #                 # last cell, width = cell start to end of image
    #                 # get width & height
    #                 width = tW
    #                 height = tH
    #             tW = width
    #             tH = height
    #
    #             # get roi (region of interest) to find an x
    #             roi = img[h[1]:h[1]+width, v[0]:v[0]+height]
    #
    #             # save image (for testing)
    #             dir = 'imgs/table%s' % (tableCnt+1)
    #             if not os.path.exists(dir):
    #                 os.makedirs(dir)
    #             fn = '%s/roi_r%s-c%s.png' % (dir, i, j)
    #             cv.imwrite(fn, roi)
    #
    #             # if roi contains an x, add x to array, else add _
    #             roi_gry = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    #             ret, thresh = cv.threshold(roi_gry, 127, 255, 0)
    #             contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #
    #             if len(contours) > 1:
    #                 # there is an x for 2 or more contours
    #                 row.append('x')
    #             else:
    #                 # there is no x when len(contours) is <= 1
    #                 row.append('_')
    #         row.pop()
    #         rows.append(row)

    # close image to eliminate text and thin lines, hence you get
    # only white blocks separated by black hough lines
    closed_binary_image = Closing(img)

    # find contours in the image
    contours, hierachy = cv.findContours(closed_binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    imgCopy = cv.drawContours(imgCopy, contours, -1, (255, 255, 255), 1)

    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # Crop the result
        final_image = imgCopy[y:y + h + 1, x:x + w + 1]
        # make directory imgs
        dir = 'imgs/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        fn = '%s/cell%s.png' % (dir, i)
        cv.imwrite(fn, final_image)



# __main__
dataset = ReadImages(r'dataset_module1')
HoughLinesManual(dataset[1][1])
cv.waitKey()
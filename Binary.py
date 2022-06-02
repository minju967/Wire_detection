
# Image Binarization
# threshold < 0 : black
# threshold > 1 : white
import collections
import os
import glob
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 이미지 로드
# color image -> gray scale 변환
# threshold 설정
# image binarization

def show(window, img):
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_circle(img, circles):
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # show('hough circle', img)


def draw_contours(img, contours, circle):
    result = False
    if contours is not None:
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            error = pow((x-circle[0]), 2) + pow((y-circle[1]), 2)
            if error < 1000:
                cv2.circle(img, center, radius, (255, 0, 255), 2)
                result = True
            else:
                # cv2.circle(img, center, radius, (0, 255, 0), 2)
                print(error)
        # show("contours", img)
    else:
        print(f'존재하는 contours 없음')

    return result

def find_circle(img_path):
    path = img_path
    img_bgr = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray)
    # plt.show()
    # show('img_gray', img_gray)

    row = img_gray.shape[0]
    col = img_gray.shape[1]
    img_bin = np.zeros((224, 224))

    count = collections.Counter(img_gray.ravel())
    threshold = max(count, key=count.get)

    if threshold == 255:
        del(count[255])
        threshold = max(count, key=count.get)

    print(f'threshold: {threshold}')

    img_bin = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Circles
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=15, minRadius=0, maxRadius=0)
    draw_circle(img_bgr, circles)

    exist_contour = False

    if circles is not None:
        circle_x = circles[0][0][0]
        circle_y = circles[0][0][1]

        # Contours
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        exist_contour = draw_contours(img_bgr, contours, (circle_x, circle_y))

    else:
        print(" 원이 존재하지 않음 ")
    if not exist_contour:
        # show("Result", img_bgr)
        destination = os.path.join(move_path, path.split('\\')[-1])
        shutil.move(path, destination)

        # show("Result", img_bgr)

dir_path = 'D:\\Data\\Wire'
move_path = 'D:\\Data\\None_wire'
img_list = glob.glob(dir_path+'./*.png')

# img_path = 'D:\\Data\\None_wire\\18318.png'
# find_circle(img_path)

for img_path in img_list:
    find_circle(img_path)



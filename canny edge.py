import numpy as np
import cv2
import matplotlib.pyplot as plt
import selectivesearch

def show(window, img):
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = 'C:\\Users\\user\\PycharmProjects\\Wire_detection\\image_sample\\91091.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Canny(img, minimum thresholding value, maximum thresholding value)
# maximum thresholding value가 낮을 수록 더 많은 edge 추출
edge2 = cv2.Canny(img, 150, 150)
# show('Canny Edge2', edge2)

# circles = cv2.HoughCircles(edge2, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)
circles = cv2.HoughCircles(edge2, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=18, minRadius=0, maxRadius=0)
# 50: 검출한 원의 중심과의 최소거리 값. 해당 거리보다 작은면 원으로 판별되지 않는다.
# param1: Canny Edge에 전달되는 인자 값
# param2: 검출 결과를 보면서 조절 => 작으면 오류가 높고 크면 검출룰이 낮아진다.
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    # show('circle', img)

selective_img = np.stack((img,)*3, axis=-1)
# show('img', selective_img)
_, regions = selectivesearch.selective_search(selective_img, scale=100, min_size=100)
cand_rects = [cand['rect'] for cand in regions if cand['size'] < 1000]
# cand_rects = [cand['rect'] for cand in regions]

green_rgb = (125, 255, 51) # bounding box color
img_rgb_copy = img.copy() # 이미지 복사

for rect in cand_rects:
    left = rect[0]
    top = rect[1]
    right = left + rect[2]
    bottom = top + rect[3]
    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=2)
    # show('result', img_rgb_copy)


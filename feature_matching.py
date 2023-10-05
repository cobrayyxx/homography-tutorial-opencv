import cv2
import numpy as np
import random
# Code is modification from https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

img1 = cv2.imread('source.png')
img2 = cv2.imread('dest.png')
img1 = cv2.resize(img1, (img2.shape[1]//2, img2.shape[0]//2), interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2), interpolation = cv2.INTER_AREA)
patternSize = (9,6)
ret1, corners1 = cv2.findChessboardCorners(img1, patternSize)
ret2, corners2 = cv2.findChessboardCorners(img2, patternSize)

H, _ = cv2.findHomography(corners1, corners2)
img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))


img_draw_matches = cv2.hconcat([img1, img2])

for i in range(len(corners1)):
    pt1 = np.array([corners1[i][0][0], corners1[i][0][1], 1])
    pt1 = pt1.reshape(3, 1)
    pt2 = np.dot(H, pt1)
    pt2 = pt2/pt2[2]
    end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.line(img_draw_matches, tuple([int(j) for j in corners1[i][0]]), end, color, 2)
cv2.resize(img_draw_matches, (80,80),interpolation = cv2.INTER_CUBIC)
cv2.imshow("Draw matches", img_draw_matches)
cv2.waitKey(0)

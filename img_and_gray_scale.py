import cv2 as cv
import numpy as np
import os

# soccer.jpg 이미지 로드
img = cv.imread('soccer.jpg')
if img is None:
    print("오류: soccer.jpg 파일을 찾을 수 없습니다.")
    exit()
    
# BGR 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 그레이스케일을 3채널로 변환하여 원본과 같은 형식으로 만들기
gray_3channel = np.dstack([gray, gray, gray])

# 이미지들을 리스트에 저장
frames = [img, gray_3channel]

# 이미지들을 나란히 이어붙히기
if len(frames) > 0:
    imgs = frames[0]
    for i in range(1, len(frames)):
        imgs = np.hstack((imgs, frames[i]))

# 결과 이미지 표시
cv.namedWindow('Original and Grayscale Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Original and Grayscale Image', imgs.shape[1], imgs.shape[0])
cv.imshow('Original and Grayscale Image', imgs)

cv.waitKey()
cv.destroyAllWindows()

print(type(img))
print(img.shape)
print("Grayscale shape:", gray.shape)
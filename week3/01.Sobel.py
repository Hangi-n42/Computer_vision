from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 현재 파이썬 파일이 있는 폴더 경로를 계산한다.
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로를 설정한다.
RESULT_DIR = SCRIPT_DIR / "results_sobel"

# 결과 폴더가 없으면 자동으로 생성한다.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 과제에서 지정한 입력 이미지 경로를 구성한다.
image_path = SCRIPT_DIR.parent.parent / "images" / "edgeDetectionImage.jpg"

# 지정한 입력 이미지가 없으면 명확한 에러 메시지와 함께 중단한다.
if not image_path.exists():
    raise FileNotFoundError(
        "입력 이미지를 찾지 못했습니다. "
        "확인 경로: C:/Projects/Computer_vision/images/edgeDetectionImage.jpg"
    )

# cv.imread()로 컬러 이미지를 불러온다.
color_bgr = cv.imread(str(image_path))

# 이미지 로드 실패 시 예외를 발생시킨다.
if color_bgr is None:
    raise RuntimeError(f"이미지 로드에 실패했습니다: {image_path}")

# cv.cvtColor()로 BGR 이미지를 그레이스케일로 변환한다.
gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)

# cv.Sobel()로 x축 방향 에지를 계산한다 (ddepth=CV_64F, dx=1, dy=0).
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# cv.Sobel()로 y축 방향 에지를 계산한다 (ddepth=CV_64F, dx=0, dy=1).
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# cv.magnitude()로 x/y 기울기를 결합해 에지 강도(크기)를 계산한다.
magnitude = cv.magnitude(sobel_x, sobel_y)

# cv.convertScaleAbs()로 x축 에지를 시각화 가능한 uint8로 변환한다.
sobel_x_abs = cv.convertScaleAbs(sobel_x)

# cv.convertScaleAbs()로 y축 에지를 시각화 가능한 uint8로 변환한다.
sobel_y_abs = cv.convertScaleAbs(sobel_y)

# cv.convertScaleAbs()로 에지 강도 이미지를 시각화 가능한 uint8로 변환한다.
magnitude_abs = cv.convertScaleAbs(magnitude)

# matplotlib 표시를 위해 BGR 원본 이미지를 RGB로 변환한다.
color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

# 중간 결과물인 x축 소벨 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "sobel_x_abs.png"), sobel_x_abs)

# 중간 결과물인 y축 소벨 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "sobel_y_abs.png"), sobel_y_abs)

# 최종 결과물인 에지 강도 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "sobel_magnitude_abs.png"), magnitude_abs)

# 원본/에지 강도 비교 시각화를 위한 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 1행 2열 중 첫 번째 위치에 원본 이미지를 배치한다.
plt.subplot(1, 2, 1)

# 원본 RGB 이미지를 화면에 그린다.
plt.imshow(color_rgb)

# 원본 이미지 제목을 설정한다.
plt.title("Original")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 1행 2열 중 두 번째 위치에 에지 강도 이미지를 배치한다.
plt.subplot(1, 2, 2)

# 에지 강도 이미지를 흑백(cmap='gray')으로 시각화한다.
plt.imshow(magnitude_abs, cmap="gray")

# 에지 강도 이미지 제목을 설정한다.
plt.title("Sobel Edge Magnitude")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 서브플롯 간 간격을 자동 조정한다.
plt.tight_layout()

# 최종 비교 시각화 이미지를 파일로 저장한다.
plt.savefig(str(RESULT_DIR / "sobel_visualization.png"), dpi=150)

# 메모리 정리를 위해 Figure를 닫는다.
plt.close()

# 결과 저장 폴더 경로를 콘솔에 출력한다.
print(f"Input image: {image_path}")

# 결과 저장 폴더 경로를 콘솔에 출력한다.
print(f"Saved results to: {RESULT_DIR}")

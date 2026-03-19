from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 현재 파이썬 파일이 있는 폴더 경로를 계산한다.
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로를 설정한다.
RESULT_DIR = SCRIPT_DIR / "results_grabcut"

# 결과 폴더가 없으면 자동으로 생성한다.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 과제 안내의 입력 경로(오탈자 포함)를 먼저 구성한다.
requested_path = SCRIPT_DIR.parent.parent / "images" / "coffe cup.JPG"

# 실제 파일명 경로를 함께 구성한다.
actual_path = SCRIPT_DIR.parent.parent / "images" / "coffee cup.JPG"

# 우선순위에 따라 사용할 입력 이미지 경로를 선택한다.
if requested_path.exists():
    # 과제 안내 경로가 존재하면 해당 파일을 사용한다.
    image_path = requested_path
elif actual_path.exists():
    # 안내 경로가 없고 실제 파일이 있으면 실제 파일을 사용한다.
    image_path = actual_path
else:
    # 두 경로 모두 없으면 에러 메시지와 함께 중단한다.
    raise FileNotFoundError(
        "입력 이미지를 찾지 못했습니다. "
        "확인 경로: C:/Projects/Computer_vision/images/coffe cup.JPG, "
        "C:/Projects/Computer_vision/images/coffee cup.JPG"
    )

# cv.imread()로 원본 컬러 이미지를 불러온다.
color_bgr = cv.imread(str(image_path))

# 이미지 로드에 실패하면 예외를 발생시킨다.
if color_bgr is None:
    raise RuntimeError(f"이미지 로드에 실패했습니다: {image_path}")

# 이미지의 높이와 너비를 추출한다.
height, width = color_bgr.shape[:2]

# GrabCut에 사용할 초기 사각형 좌표를 설정한다.
rect_x = int(width * 0.08)

# GrabCut에 사용할 초기 사각형 좌표를 설정한다.
rect_y = int(height * 0.08)

# GrabCut에 사용할 초기 사각형 너비를 설정한다.
rect_w = int(width * 0.84)

# GrabCut에 사용할 초기 사각형 높이를 설정한다.
rect_h = int(height * 0.84)

# GrabCut의 초기 사각형을 (x, y, w, h) 형식으로 구성한다.
rect = (rect_x, rect_y, rect_w, rect_h)

# GrabCut 상태를 저장할 마스크 배열을 0으로 초기화한다.
mask = np.zeros((height, width), np.uint8)

# 배경 모델을 힌트대로 (1,65) float64 0배열로 초기화한다.
bgd_model = np.zeros((1, 65), np.float64)

# 전경 모델을 힌트대로 (1,65) float64 0배열로 초기화한다.
fgd_model = np.zeros((1, 65), np.float64)

# cv.grabCut()을 사각형 초기화 모드로 실행한다.
cv.grabCut(
    color_bgr,            # 분할 대상 원본 이미지를 전달한다.
    mask,                 # 분할 라벨을 저장할 마스크를 전달한다.
    rect,                 # 사용자가 지정한 초기 사각형 영역을 전달한다.
    bgd_model,            # 배경 GMM 모델 버퍼를 전달한다.
    fgd_model,            # 전경 GMM 모델 버퍼를 전달한다.
    5,                    # 반복 횟수를 5회로 설정한다.
    cv.GC_INIT_WITH_RECT, # 사각형 기반 초기화 모드를 지정한다.
)

# np.where()로 확정/가능 전경만 1, 나머지는 0인 바이너리 마스크를 만든다.
foreground_mask = np.where(
    (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
    1,
    0,
).astype("uint8")

# 바이너리 마스크를 0/255 형태로 시각화용 변환한다.
mask_vis = foreground_mask * 255

# 원본 이미지에 바이너리 마스크를 곱해 배경을 제거한다.
extracted_bgr = color_bgr * foreground_mask[:, :, np.newaxis]

# 중간 결과물인 GrabCut 마스크 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "grabcut_mask.png"), mask_vis)

# 최종 결과물인 배경 제거 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "grabcut_extracted.png"), extracted_bgr)

# 사각형 영역을 보여주기 위해 원본 복사본 이미지를 생성한다.
rect_preview = color_bgr.copy()

# 초기 사각형 영역을 녹색 선으로 표시한다.
cv.rectangle(rect_preview, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

# 사각형 미리보기 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "grabcut_rect_preview.png"), rect_preview)

# Matplotlib 표시를 위해 BGR 원본을 RGB로 변환한다.
original_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

# Matplotlib 표시를 위해 BGR 추출 결과를 RGB로 변환한다.
extracted_rgb = cv.cvtColor(extracted_bgr, cv.COLOR_BGR2RGB)

# 원본/마스크/추출 결과를 나란히 보여줄 Figure를 생성한다.
plt.figure(figsize=(15, 5))

# 1행 3열 중 첫 번째 위치에 원본 이미지를 배치한다.
plt.subplot(1, 3, 1)

# 원본 RGB 이미지를 화면에 표시한다.
plt.imshow(original_rgb)

# 원본 이미지 제목을 설정한다.
plt.title("Original")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 1행 3열 중 두 번째 위치에 마스크 이미지를 배치한다.
plt.subplot(1, 3, 2)

# 마스크 이미지를 흑백(cmap='gray')으로 시각화한다.
plt.imshow(mask_vis, cmap="gray")

# 마스크 이미지 제목을 설정한다.
plt.title("GrabCut Mask")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 1행 3열 중 세 번째 위치에 배경 제거 이미지를 배치한다.
plt.subplot(1, 3, 3)

# 배경 제거 RGB 이미지를 화면에 표시한다.
plt.imshow(extracted_rgb)

# 배경 제거 이미지 제목을 설정한다.
plt.title("Background Removed")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 서브플롯 간 간격을 자동으로 조정한다.
plt.tight_layout()

# 3분할 시각화 결과를 파일로 저장한다.
plt.savefig(str(RESULT_DIR / "grabcut_visualization.png"), dpi=150)

# 메모리 정리를 위해 Figure를 닫는다.
plt.close()

# 전체 픽셀 수를 계산한다.
total_pixels = int(height * width)

# 전경으로 분류된 픽셀 수를 계산한다.
foreground_pixels = int(np.count_nonzero(foreground_mask))

# 전경 픽셀 비율(%)을 계산한다.
foreground_ratio = (foreground_pixels / total_pixels) * 100.0

# 사용한 입력 이미지 경로를 출력한다.
print(f"Input image: {image_path}")

# GrabCut 초기 사각형 좌표를 출력한다.
print(f"Initial rectangle (x, y, w, h): {rect}")

# 전경 픽셀 수를 출력한다.
print(f"Foreground pixels: {foreground_pixels}")

# 전경 픽셀 비율을 출력한다.
print(f"Foreground ratio: {foreground_ratio:.2f}%")

# 결과 저장 폴더 경로를 출력한다.
print(f"Saved results to: {RESULT_DIR}")

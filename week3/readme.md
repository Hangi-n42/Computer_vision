# Week3

## 01.소벨 에지 검출 및 결과 시각화

### 과제 설명
- edgeDetectionImage에 해당하는 입력 이미지를 그레이스케일로 변환한다.
- Sobel 필터를 사용하여 x축, y축 방향 기울기 에지를 각각 계산한다.
- `cv.magnitude()`를 사용해 두 방향 기울기를 결합한 에지 강도 이미지를 만든다.
- Matplotlib로 원본 이미지와 에지 강도 이미지를 나란히 시각화한다.

### 중간 결과물
- `week3/results_sobel/sobel_x_abs.png`
  - x축 방향 소벨 결과(세로 경계에 더 민감).
- `week3/results_sobel/sobel_y_abs.png`
  - y축 방향 소벨 결과(가로 경계에 더 민감).

### 최종 결과물
- `week3/results_sobel/sobel_magnitude_abs.png`
  - x/y 기울기를 결합한 최종 에지 강도 이미지.
- `week3/results_sobel/sobel_visualization.png`
  - 원본 이미지와 최종 에지 강도 이미지를 나란히 배치한 시각화 결과.

### 코드 (주석 포함)
```python
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

# 과제 안내에 적힌 파일명(오탈자 포함) 경로를 먼저 구성한다.
requested_path = SCRIPT_DIR.parent.parent / "images" / "coffe cup.JPG"

# 실제 폴더에 존재하는 파일명 경로를 함께 구성한다.
actual_path = SCRIPT_DIR.parent.parent / "images" / "coffee cup.JPG"

# 우선순위에 따라 사용할 입력 이미지 경로를 선택한다.
if requested_path.exists():
    # 과제 안내 경로가 존재하면 해당 파일을 사용한다.
    image_path = requested_path
elif actual_path.exists():
    # 안내 경로가 없고 실제 파일이 있으면 실제 파일을 사용한다.
    image_path = actual_path
else:
    # 두 경로 모두 없으면 명확한 에러 메시지와 함께 중단한다.
    raise FileNotFoundError(
        "입력 이미지를 찾지 못했습니다. "
        "확인 경로: C:/Projects/Computer_vision/images/coffe cup.JPG, "
        "C:/Projects/Computer_vision/images/coffee cup.JPG"
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
```

## 02.캐니 에지 및 허프 변환을 이용한 직선 검출

### 과제 설명
- 입력 이미지(`dabo`)에서 캐니 에지 검출로 에지 맵을 생성한다.
- 생성된 에지 맵에 확률적 허프 변환을 적용해 직선을 검출한다.
- 검출된 직선을 원본 이미지 위에 빨간색(`(0, 0, 255)`)으로 표시한다.
- Matplotlib를 이용해 원본 이미지와 직선이 그려진 이미지를 나란히 시각화한다.

### 중간 결과물
- `week3/results_canny_hough/canny_edges.png`
  - 캐니 에지 검출 결과로, 허프 변환의 입력으로 사용되는 이진 에지 맵.

### 최종 결과물
- `week3/results_canny_hough/hough_lines_overlay.png`
  - 원본 이미지 위에 검출 직선을 빨간색으로 그린 최종 검출 이미지.
- `week3/results_canny_hough/canny_hough_visualization.png`
  - 원본 이미지와 직선 검출 결과를 좌우로 배치한 비교 시각화 이미지.
  
- 콘솔 출력 결과:
```text
Input image: C:\Projects\Computer_vision\images\dabo.JPG
Detected lines: 680
Saved results to: C:\Projects\Computer_vision\Computer_vision\week3\results_canny_hough
```

### 코드 (주석 포함)
```python
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 현재 파이썬 파일이 있는 폴더 경로를 계산한다.
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로를 설정한다.
RESULT_DIR = SCRIPT_DIR / "results_canny_hough"

# 결과 폴더가 없으면 자동으로 생성한다.
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 과제 안내에 적힌 입력 경로(대문자 확장자 포함)를 먼저 구성한다.
requested_path = SCRIPT_DIR.parent.parent / "images" / "dabo.JPG"

# 실제 폴더에 존재하는 파일명(소문자 확장자) 경로를 함께 구성한다.
actual_path = SCRIPT_DIR.parent.parent / "images" / "dabo.jpg"

# 우선순위에 따라 사용할 입력 이미지 경로를 선택한다.
if requested_path.exists():
  # 과제 안내 경로가 존재하면 해당 파일을 사용한다.
  image_path = requested_path
elif actual_path.exists():
  # 안내 경로가 없고 실제 파일이 있으면 실제 파일을 사용한다.
  image_path = actual_path
else:
  # 두 경로 모두 없으면 명확한 에러 메시지와 함께 중단한다.
  raise FileNotFoundError(
    "입력 이미지를 찾지 못했습니다. "
    "확인 경로: C:/Projects/Computer_vision/images/dabo.JPG, "
    "C:/Projects/Computer_vision/images/dabo.jpg"
  )

# cv.imread()로 원본 컬러 이미지를 불러온다.
color_bgr = cv.imread(str(image_path))

# 이미지 로드에 실패하면 예외를 발생시킨다.
if color_bgr is None:
  raise RuntimeError(f"이미지 로드에 실패했습니다: {image_path}")

# 캐니 에지를 적용하기 위해 원본 이미지를 그레이스케일로 변환한다.
gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)

# cv.Canny()로 에지 맵을 생성한다 (threshold1=100, threshold2=200).
edges = cv.Canny(gray, 100, 200)

# 허프 직선 검출을 위해 확률적 허프 변환을 수행한다.
lines = cv.HoughLinesP(
  edges,               # 캐니 에지 결과를 입력으로 사용한다.
  rho=1,               # 거리 해상도는 1픽셀로 설정한다.
  theta=np.pi / 180,   # 각도 해상도는 1도로 설정한다.
  threshold=80,        # 누적값 임계치를 설정한다.
  minLineLength=50,    # 최소 직선 길이를 설정한다.
  maxLineGap=10,       # 같은 직선으로 연결할 최대 간격을 설정한다.
)

# 원본 위에 검출 직선을 그리기 위해 복사본 이미지를 만든다.
line_overlay = color_bgr.copy()

# 검출된 직선 개수를 저장할 변수를 초기화한다.
line_count = 0

# 직선이 하나 이상 검출된 경우에만 반복문을 실행한다.
if lines is not None:
  # 검출된 각 직선을 순회한다.
  for line in lines:
    # HoughLinesP 결과에서 직선의 양 끝점 좌표를 추출한다.
    x1, y1, x2, y2 = line[0]

    # cv.line()으로 원본 복사본 위에 빨간색 직선을 그린다.
    cv.line(line_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 검출된 직선 개수를 1 증가시킨다.
    line_count += 1

# Matplotlib 표시를 위해 원본 BGR 이미지를 RGB로 변환한다.
color_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

# Matplotlib 표시를 위해 직선 오버레이 BGR 이미지를 RGB로 변환한다.
line_overlay_rgb = cv.cvtColor(line_overlay, cv.COLOR_BGR2RGB)

# 중간 결과물인 캐니 에지 맵을 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "canny_edges.png"), edges)

# 최종 결과물인 허프 직선 오버레이 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "hough_lines_overlay.png"), line_overlay)

# 원본/직선 검출 결과 나란히 시각화를 위한 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 1행 2열 중 첫 번째 위치에 원본 이미지를 배치한다.
plt.subplot(1, 2, 1)

# 원본 이미지를 화면에 표시한다.
plt.imshow(color_rgb)

# 원본 이미지 제목을 설정한다.
plt.title("Original")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 1행 2열 중 두 번째 위치에 직선 검출 결과 이미지를 배치한다.
plt.subplot(1, 2, 2)

# 직선이 그려진 이미지를 화면에 표시한다.
plt.imshow(line_overlay_rgb)

# 직선 검출 결과 제목을 설정한다.
plt.title("Canny + Hough Lines")

# 축 눈금을 숨겨 시각적 잡음을 줄인다.
plt.axis("off")

# 서브플롯 간격을 자동 조정한다.
plt.tight_layout()

# 나란히 시각화한 최종 이미지를 파일로 저장한다.
plt.savefig(str(RESULT_DIR / "canny_hough_visualization.png"), dpi=150)

# 메모리 정리를 위해 Figure를 닫는다.
plt.close()

# 사용한 입력 이미지 경로를 콘솔에 출력한다.
print(f"Input image: {image_path}")

# 검출된 직선 개수를 콘솔에 출력한다.
print(f"Detected lines: {line_count}")

# 결과 저장 폴더 경로를 콘솔에 출력한다.
print(f"Saved results to: {RESULT_DIR}")
```


## 03.GrabCut을 이용한 대화식 영역 분할 및 객체 추출

### 과제 설명
- `coffee cup` 이미지를 대상으로 사용자가 지정한 사각형 영역을 기반으로 GrabCut 분할을 수행한다.
- GrabCut 결과를 마스크 형태로 시각화해 전경/배경 분할 상태를 확인한다.
- 마스크를 이용해 원본 이미지에서 배경을 제거하고 객체만 남긴다.
- Matplotlib로 원본 이미지, 마스크 이미지, 배경 제거 이미지를 나란히 시각화한다.

### 중간 결과물
- `week3/results_grabcut/grabcut_rect_preview.png`
  - GrabCut 초기화에 사용된 사각형 영역을 원본 이미지 위에 표시한 미리보기 이미지.
- `week3/results_grabcut/grabcut_mask.png`
  - GrabCut 결과에서 전경을 255, 배경을 0으로 나타낸 바이너리 마스크 이미지.

### 최종 결과물
- `week3/results_grabcut/grabcut_extracted.png`
  - 마스크를 원본 이미지에 적용해 배경을 제거하고 객체만 남긴 최종 추출 이미지.
- `week3/results_grabcut/grabcut_visualization.png`
  - 원본, 마스크, 배경 제거 결과를 1행 3열로 나란히 배치한 최종 시각화 이미지.

- 콘솔 출력 결과:
```text
Input image: C:\Projects\Computer_vision\images\coffee cup.JPG
Initial rectangle (x, y, w, h): (102, 76, 1075, 806)
Foreground pixels: 400407
Foreground ratio: 32.59%
Saved results to: C:\Projects\Computer_vision\Computer_vision\week3\results_grabcut
```

### 코드 (주석 포함)
```python
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
```

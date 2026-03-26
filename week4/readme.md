# Week4

## 01. SIFT를 이용한 특징점 검출 및 시각화

### 과제 설명
- 입력 이미지 `images/mot_color70.jpg`에서 SIFT 알고리즘으로 특징점을 검출한다.
- `cv.SIFT_create()`로 SIFT 객체를 생성하고, `detectAndCompute()`로 특징점과 기술자를 계산한다.
- `cv.drawKeypoints()`로 특징점을 이미지 위에 시각화한다.
- Matplotlib로 원본 이미지와 특징점 시각화 이미지를 나란히 비교한다.
- `flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`를 사용해 특징점의 방향/크기 정보를 함께 표시한다.

### 중간 결과물 (설명 포함)
- `week4/results_sift/sift_keypoints_simple.jpg`
  - 점 형태 중심의 단순 특징점 시각화 이미지.
- `week4/results_sift/sift_summary.txt`
  - 입력 이미지 경로, 검출 특징점 개수, 기술자 행렬 크기를 기록한 텍스트 요약.

### 최종 결과물 (설명 포함)
- `week4/results_sift/sift_keypoints_rich.jpg`
  - 특징점의 방향과 스케일(크기)까지 표시한 최종 특징점 시각화 이미지.
- `week4/results_sift/sift_visualization.png`
  - 원본 이미지와 Rich 특징점 이미지를 좌우로 배치한 최종 비교 시각화.
- 콘솔 출력 결과:
```text
Input image: C:\Projects\Computer_vision\images\mot_color70.jpg
Detected keypoints: 400
Descriptor shape: (400, 128)
Saved results to: C:\Projects\Computer_vision\Computer_vision\week4\results_sift
```

### 코드 (주석 포함)
```python
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results_sift"
INPUT_IMAGE_PATH = SCRIPT_DIR.parent.parent / "images" / "mot_color70.jpg"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_IMAGE_PATH.exists():
    raise FileNotFoundError(
        "입력 이미지를 찾지 못했습니다. "
        "확인 경로: C:/Projects/Computer_vision/images/mot_color70.jpg"
    )

# OpenCV로 원본 컬러 이미지를 불러온다.
color_bgr = cv.imread(str(INPUT_IMAGE_PATH))

# 이미지 로드 실패 시 즉시 중단한다.
if color_bgr is None:
    raise RuntimeError(f"이미지 로드에 실패했습니다: {INPUT_IMAGE_PATH}")

# SIFT는 그레이스케일 기반으로 동작하므로 회색조로 변환한다.
gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)

# 특징점이 너무 많아지지 않도록 nfeatures를 제한해 SIFT 객체를 생성한다.
sift = cv.SIFT_create(nfeatures=400)

# SIFT로 특징점(keypoints)과 기술자(descriptors)를 함께 계산한다.
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 단순 점 형태로 특징점을 그린 중간 결과 이미지를 생성한다.
keypoints_simple_bgr = cv.drawKeypoints(
    color_bgr,
    keypoints,
    None,
    color=(0, 255, 0),
)

# 크기/방향 정보까지 보이도록 Rich Keypoint 플래그로 최종 결과 이미지를 생성한다.
keypoints_rich_bgr = cv.drawKeypoints(
    color_bgr,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# matplotlib 표시용으로 BGR 이미지를 RGB로 변환한다.
original_rgb = cv.cvtColor(color_bgr, cv.COLOR_BGR2RGB)

# matplotlib 표시용으로 단순 특징점 이미지를 RGB로 변환한다.
keypoints_simple_rgb = cv.cvtColor(keypoints_simple_bgr, cv.COLOR_BGR2RGB)

# matplotlib 표시용으로 Rich 특징점 이미지를 RGB로 변환한다.
keypoints_rich_rgb = cv.cvtColor(keypoints_rich_bgr, cv.COLOR_BGR2RGB)

# 중간 결과물인 단순 특징점 시각화 이미지를 저장한다.
cv.imwrite(str(RESULT_DIR / "sift_keypoints_simple.jpg"), keypoints_simple_bgr)

# 최종 결과물인 Rich 특징점 시각화 이미지를 저장한다.
cv.imwrite(str(RESULT_DIR / "sift_keypoints_rich.jpg"), keypoints_rich_bgr)

# 원본/최종 결과를 나란히 보여줄 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 1행 2열 중 첫 번째 위치에 원본 이미지를 배치한다.
plt.subplot(1, 2, 1)

# 원본 이미지를 표시한다.
plt.imshow(original_rgb)

# 원본 이미지 제목을 설정한다.
plt.title("Original")

# 축 눈금을 숨긴다.
plt.axis("off")

# 1행 2열 중 두 번째 위치에 SIFT 특징점 이미지를 배치한다.
plt.subplot(1, 2, 2)

# Rich 특징점 결과 이미지를 표시한다.
plt.imshow(keypoints_rich_rgb)

# 특징점 결과 제목을 설정한다.
plt.title("SIFT Keypoints (Rich)")

# 축 눈금을 숨긴다.
plt.axis("off")

# 레이아웃을 자동 정렬한다.
plt.tight_layout()

# 나란히 비교한 최종 시각화 이미지를 저장한다.
plt.savefig(str(RESULT_DIR / "sift_visualization.png"), dpi=150)

# 메모리 정리를 위해 Figure를 닫는다.
plt.close()

# 특징점 개수와 기술자 행렬 크기를 텍스트로 정리한다.
summary_lines = [
    f"Input image: {INPUT_IMAGE_PATH}",
    f"Detected keypoints: {len(keypoints)}",
]

# 기술자 계산이 성공한 경우 shape 정보를 함께 기록한다.
if descriptors is not None:
    summary_lines.append(f"Descriptor shape: {descriptors.shape}")
else:
    summary_lines.append("Descriptor shape: None")

# 결과 요약 텍스트 파일을 저장한다.
(RESULT_DIR / "sift_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# 콘솔에 입력 이미지 경로를 출력한다.
print(f"Input image: {INPUT_IMAGE_PATH}")

# 콘솔에 검출 특징점 개수를 출력한다.
print(f"Detected keypoints: {len(keypoints)}")

# 콘솔에 기술자 행렬 정보를 출력한다.
print(f"Descriptor shape: {None if descriptors is None else descriptors.shape}")

# 콘솔에 결과 저장 경로를 출력한다.
print(f"Saved results to: {RESULT_DIR}")
```

## 02. SIFT를 이용한 두 영상 간 특징점 매칭

### 과제 설명
- 두 개의 이미지(`mot_color70.jpg`, `mot_color80.jpg`)를 입력으로 받아 SIFT 특징점 기반 매칭을 수행한다.
- 각 이미지에서 `cv.SIFT_create()`와 `detectAndCompute()`로 특징점/기술자를 추출한다.
- `cv.BFMatcher()`를 사용해 두 영상의 특징점을 매칭한다.
- `knnMatch()`와 거리 비율 테스트(Lowe ratio test)를 적용해 매칭 정확도를 높인다.
- `cv.drawMatches()`로 매칭 결과를 시각화하고, Matplotlib로 결과를 출력한다.
- 현재 워크스페이스에는 `mot_color80.jpg`가 없어, 코드에서는 해당 경로를 우선 시도한 뒤 실제 존재하는 `mot_color83.jpg`를 대체 사용하도록 처리했다.

### 중간 결과물 (설명 포함)
- `week4/results_sift_match/sift_match_preview.jpg`
  - 비율 테스트를 통과한 매칭 중 상위 120개만 표시한 중간 점검용 이미지.
- `week4/results_sift_match/sift_match_summary.txt`
  - 입력 경로, 특징점 개수, KNN 쌍 수, 최종 매칭 수를 기록한 요약 텍스트.

### 최종 결과물 (설명 포함)
- `week4/results_sift_match/sift_match_final.jpg`
  - 비율 테스트를 통과한 전체 좋은 매칭 결과를 표시한 최종 매칭 이미지.
- `week4/results_sift_match/sift_match_visualization.png`
  - 중간 결과(Top 120)와 최종 결과(전체 good matches)를 나란히 비교한 시각화.
- 콘솔 출력 결과:
```text
Image1: C:\Projects\Computer_vision\images\mot_color70.jpg
Image2: C:\Projects\Computer_vision\images\mot_color83.jpg
Keypoints image1: 500
Keypoints image2: 500
Good matches: 124
Saved results to: C:\Projects\Computer_vision\Computer_vision\week4\results_sift_match
```

### 코드 (주석 포함)
```python
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results_sift_match"
IMAGE1_PATH = SCRIPT_DIR.parent.parent / "images" / "mot_color70.jpg"
REQUESTED_IMAGE2_PATH = SCRIPT_DIR.parent.parent / "images" / "mot_color80.jpg"
FALLBACK_IMAGE2_PATH = SCRIPT_DIR.parent.parent / "images" / "mot_color83.jpg"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE1_PATH.exists():
  raise FileNotFoundError(
    "첫 번째 입력 이미지를 찾지 못했습니다. "
    "확인 경로: C:/Projects/Computer_vision/images/mot_color70.jpg"
  )

if REQUESTED_IMAGE2_PATH.exists():
  IMAGE2_PATH = REQUESTED_IMAGE2_PATH
elif FALLBACK_IMAGE2_PATH.exists():
  IMAGE2_PATH = FALLBACK_IMAGE2_PATH
else:
  raise FileNotFoundError(
    "두 번째 입력 이미지를 찾지 못했습니다. "
    "확인 경로: C:/Projects/Computer_vision/images/mot_color80.jpg, "
    "C:/Projects/Computer_vision/images/mot_color83.jpg"
  )

# 두 입력 이미지를 OpenCV로 컬러 형태로 불러온다.
img1_bgr = cv.imread(str(IMAGE1_PATH))

# 두 입력 이미지를 OpenCV로 컬러 형태로 불러온다.
img2_bgr = cv.imread(str(IMAGE2_PATH))

# 이미지 로드 실패 시 즉시 중단한다.
if img1_bgr is None or img2_bgr is None:
  raise RuntimeError("입력 이미지 로드에 실패했습니다.")

# SIFT 추출을 위해 첫 번째 이미지를 그레이스케일로 변환한다.
gray1 = cv.cvtColor(img1_bgr, cv.COLOR_BGR2GRAY)

# SIFT 추출을 위해 두 번째 이미지를 그레이스케일로 변환한다.
gray2 = cv.cvtColor(img2_bgr, cv.COLOR_BGR2GRAY)

# SIFT 객체를 생성한다.
sift = cv.SIFT_create(nfeatures=500)

# 첫 번째 이미지의 특징점과 기술자를 계산한다.
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)

# 두 번째 이미지의 특징점과 기술자를 계산한다.
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 기술자 계산이 실패한 경우 예외를 발생시킨다.
if descriptors1 is None or descriptors2 is None:
  raise RuntimeError("SIFT 기술자 계산에 실패했습니다.")

# BFMatcher를 L2 거리 기반으로 생성한다.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# knnMatch(k=2)로 최근접/차근접 이웃 매칭을 계산한다.
knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Lowe 비율 테스트를 위한 기준값을 설정한다.
ratio_threshold = 0.75

# 비율 테스트를 통과한 좋은 매칭을 저장할 리스트를 만든다.
good_matches = []

# 각 기술자 쌍의 최근접/차근접 매칭을 순회한다.
for pair in knn_matches:
  # knn 결과가 2개 미만이면 건너뛴다.
  if len(pair) < 2:
    continue

  # 최근접 매칭과 차근접 매칭을 분리한다.
  m, n = pair

  # 최근접 거리 비율이 기준보다 작으면 좋은 매칭으로 채택한다.
  if m.distance < ratio_threshold * n.distance:
    good_matches.append(m)

# 매칭 결과를 거리 오름차순으로 정렬한다.
good_matches = sorted(good_matches, key=lambda d: d.distance)

# 중간 결과 시각화를 위해 상위 120개 매칭만 선택한다.
preview_matches = good_matches[:120]

# drawMatches로 중간 결과(일부 매칭) 이미지를 생성한다.
match_preview_bgr = cv.drawMatches(
  img1_bgr,
  keypoints1,
  img2_bgr,
  keypoints2,
  preview_matches,
  None,
  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# drawMatches로 최종 결과(좋은 매칭 전체) 이미지를 생성한다.
match_final_bgr = cv.drawMatches(
  img1_bgr,
  keypoints1,
  img2_bgr,
  keypoints2,
  good_matches,
  None,
  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# 중간 결과 매칭 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "sift_match_preview.jpg"), match_preview_bgr)

# 최종 결과 매칭 이미지를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "sift_match_final.jpg"), match_final_bgr)

# Matplotlib 시각화를 위해 중간 결과를 RGB로 변환한다.
match_preview_rgb = cv.cvtColor(match_preview_bgr, cv.COLOR_BGR2RGB)

# Matplotlib 시각화를 위해 최종 결과를 RGB로 변환한다.
match_final_rgb = cv.cvtColor(match_final_bgr, cv.COLOR_BGR2RGB)

# 중간/최종 매칭 결과를 비교할 Figure를 생성한다.
plt.figure(figsize=(14, 6))

# 1행 2열 중 첫 번째 위치에 중간 결과를 배치한다.
plt.subplot(1, 2, 1)

# 중간 결과 이미지를 표시한다.
plt.imshow(match_preview_rgb)

# 중간 결과 제목을 설정한다.
plt.title("SIFT Matches Preview (Top 120)")

# 축 눈금을 숨긴다.
plt.axis("off")

# 1행 2열 중 두 번째 위치에 최종 결과를 배치한다.
plt.subplot(1, 2, 2)

# 최종 결과 이미지를 표시한다.
plt.imshow(match_final_rgb)

# 최종 결과 제목을 설정한다.
plt.title("SIFT Matches Final (Ratio Test)")

# 축 눈금을 숨긴다.
plt.axis("off")

# 자동 레이아웃 정렬을 수행한다.
plt.tight_layout()

# 비교 시각화 이미지를 파일로 저장한다.
plt.savefig(str(RESULT_DIR / "sift_match_visualization.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# 매칭 요약 정보를 텍스트로 정리한다.
summary_lines = [
  f"Image1: {IMAGE1_PATH}",
  f"Image2: {IMAGE2_PATH}",
  f"Keypoints image1: {len(keypoints1)}",
  f"Keypoints image2: {len(keypoints2)}",
  f"KNN pairs: {len(knn_matches)}",
  f"Good matches (ratio<{ratio_threshold}): {len(good_matches)}",
]

# 매칭 요약 파일을 저장한다.
(RESULT_DIR / "sift_match_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# 콘솔에 첫 번째 입력 이미지 경로를 출력한다.
print(f"Image1: {IMAGE1_PATH}")

# 콘솔에 두 번째 입력 이미지 경로를 출력한다.
print(f"Image2: {IMAGE2_PATH}")

# 콘솔에 첫 번째 이미지 특징점 개수를 출력한다.
print(f"Keypoints image1: {len(keypoints1)}")

# 콘솔에 두 번째 이미지 특징점 개수를 출력한다.
print(f"Keypoints image2: {len(keypoints2)}")

# 콘솔에 좋은 매칭 개수를 출력한다.
print(f"Good matches: {len(good_matches)}")

# 콘솔에 결과 저장 경로를 출력한다.
print(f"Saved results to: {RESULT_DIR}")
```
## 03. 호모그래피를 이용한 이미지 정합

### 과제 설명
- 입력 이미지 `img1.jpg`, `img2.jpg`에서 SIFT 특징점을 검출하고 두 영상 간 대응점을 찾는다.
- `cv.BFMatcher()`와 `knnMatch(k=2)`를 사용해 특징점을 매칭하고, Lowe ratio test로 좋은 매칭점만 선별한다.
- 선별된 매칭점으로 `cv.findHomography(..., cv.RANSAC, ...)`를 수행해 이상점 영향을 줄인 호모그래피 행렬을 계산한다.
- 계산된 호모그래피로 `cv.warpPerspective()`를 적용해 두 번째 이미지를 첫 번째 이미지 좌표계로 정렬한다.
- 출력 크기는 `(w1 + w2, max(h1, h2))`로 설정하여 파노라마 형태로 정렬 결과를 확인한다.
- 특징점 매칭 결과와 정렬 결과를 나란히 시각화하여 정합 품질을 비교한다.

### 중간 결과물 (설명 포함)
- `week4/results_homography/homography_matching_preview.jpg`
  - 좋은 매칭점 중 상위 120개만 시각화한 중간 점검 이미지.
- `week4/results_homography/homography_matching_inliers.jpg`
  - RANSAC inlier로 판정된 매칭만 표시한 정제 매칭 결과.
- `week4/results_homography/homography_warped.jpg`
  - `img2.jpg`를 호모그래피로 투영(warp)한 중간 결과 이미지.
- `week4/results_homography/homography_summary.txt`
  - 특징점 수, 좋은 매칭 수, inlier 수, 호모그래피 행렬을 기록한 텍스트 요약.

### 최종 결과물 (설명 포함)
- `week4/results_homography/homography_aligned.jpg`
  - warp된 이미지와 기준 이미지를 같은 좌표계에서 정렬/합성한 최종 결과.
- `week4/results_homography/homography_visualization.png`
  - 좌측에 매칭 결과(Top 120), 우측에 정렬 결과(Aligned)를 배치한 최종 보고용 시각화.
- 콘솔 출력 결과:
```text
Image1: C:\Projects\Computer_vision\images\img1.jpg
Image2: C:\Projects\Computer_vision\images\img2.jpg
Keypoints image1: 800
Keypoints image2: 800
Good matches: 230
Inlier matches (RANSAC): 219
Homography matrix:
[[ 6.14887333e-01  2.85398893e-02  2.53569588e+02]
 [-1.05163895e-01  8.68326026e-01  2.26303583e+01]
 [-5.34752075e-04  2.34241583e-06  1.00000000e+00]]
Saved results to: C:\Projects\Computer_vision\Computer_vision\week4\results_homography
```

### 코드 (주석 포함)
```python
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results_homography"
IMAGE1_PATH = SCRIPT_DIR.parent.parent / "images" / "img1.jpg"
IMAGE2_PATH = SCRIPT_DIR.parent.parent / "images" / "img2.jpg"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 과제에서 지정한 첫 번째 입력 이미지 존재 여부를 확인한다.
if not IMAGE1_PATH.exists():
  # 지정 이미지가 없으면 오류를 발생시켜 즉시 종료한다.
  raise FileNotFoundError(f"입력 이미지를 찾지 못했습니다: {IMAGE1_PATH}")

# 과제에서 지정한 두 번째 입력 이미지 존재 여부를 확인한다.
if not IMAGE2_PATH.exists():
  # 지정 이미지가 없으면 오류를 발생시켜 즉시 종료한다.
  raise FileNotFoundError(f"입력 이미지를 찾지 못했습니다: {IMAGE2_PATH}")

# 첫 번째 이미지를 컬러(BGR)로 불러온다.
img1_bgr = cv.imread(str(IMAGE1_PATH))

# 두 번째 이미지를 컬러(BGR)로 불러온다.
img2_bgr = cv.imread(str(IMAGE2_PATH))

# 이미지 로드가 실패한 경우를 방어적으로 점검한다.
if img1_bgr is None or img2_bgr is None:
  # 로드 실패 시 다음 단계 계산이 무의미하므로 종료한다.
  raise RuntimeError("입력 이미지 로드에 실패했습니다.")

# SIFT는 그레이스케일 입력을 사용하므로 첫 번째 이미지를 변환한다.
gray1 = cv.cvtColor(img1_bgr, cv.COLOR_BGR2GRAY)

# SIFT는 그레이스케일 입력을 사용하므로 두 번째 이미지를 변환한다.
gray2 = cv.cvtColor(img2_bgr, cv.COLOR_BGR2GRAY)

# SIFT 추출기를 생성한다.
sift = cv.SIFT_create(nfeatures=800)

# 첫 번째 이미지의 특징점과 기술자를 계산한다.
kp1, des1 = sift.detectAndCompute(gray1, None)

# 두 번째 이미지의 특징점과 기술자를 계산한다.
kp2, des2 = sift.detectAndCompute(gray2, None)

# 기술자 생성 실패를 확인한다.
if des1 is None or des2 is None:
  # 기술자가 없으면 매칭 자체가 불가능하므로 종료한다.
  raise RuntimeError("SIFT 기술자 생성에 실패했습니다.")

# L2 거리 기반 BFMatcher를 생성한다.
matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# 각 특징점에 대해 최근접 이웃 2개를 찾는다.
knn_matches = matcher.knnMatch(des1, des2, k=2)

# Lowe ratio test를 통과한 좋은 매칭만 저장할 리스트를 만든다.
good_matches = []

# KNN으로 얻은 매칭 쌍을 순회한다.
for pair in knn_matches:
  # 이웃이 2개 미만이면 ratio test를 수행할 수 없어 건너뛴다.
  if len(pair) < 2:
    # 다음 매칭 쌍으로 넘어간다.
    continue

  # 가장 가까운 매칭과 두 번째 매칭을 꺼낸다.
  m, n = pair

  # 첫 번째 거리가 두 번째 거리의 0.7배 미만이면 좋은 매칭으로 채택한다.
  if m.distance < 0.7 * n.distance:
    # 조건을 만족한 매칭을 결과 리스트에 추가한다.
    good_matches.append(m)

# 호모그래피 계산 최소 조건(4개 매칭점)을 확인한다.
if len(good_matches) < 4:
  # 매칭점이 부족하면 호모그래피를 구할 수 없으므로 종료한다.
  raise RuntimeError(f"좋은 매칭점이 부족합니다: {len(good_matches)}")

# img2 좌표계를 원본으로 사용하기 위해 trainIdx 기준 좌표를 모은다.
src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# img1 좌표계를 목표로 사용하기 위해 queryIdx 기준 좌표를 모은다.
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC을 사용해 이상점 영향이 줄어든 호모그래피를 계산한다.
H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 호모그래피 계산 결과 유효성을 점검한다.
if H is None or inlier_mask is None:
  # 행렬 또는 마스크가 없으면 정합 결과를 신뢰할 수 없어 종료한다.
  raise RuntimeError("호모그래피 계산에 실패했습니다.")

# Inlier 마스크를 1차원 리스트로 변환한다.
inlier_flags = inlier_mask.ravel().tolist()

# RANSAC으로 살아남은 inlier 매칭만 분리한다.
inlier_matches = [m for m, flag in zip(good_matches, inlier_flags) if flag]

# 두 이미지의 크기를 읽어 파노라마 캔버스 크기를 계산한다.
h1, w1 = img1_bgr.shape[:2]

# 두 이미지의 크기를 읽어 파노라마 캔버스 크기를 계산한다.
h2, w2 = img2_bgr.shape[:2]

# 문제 조건에 맞게 출력 폭은 w1+w2로 설정한다.
panorama_width = w1 + w2

# 문제 조건에 맞게 출력 높이는 max(h1, h2)로 설정한다.
panorama_height = max(h1, h2)

# img2를 img1 좌표계로 투영하여 파노라마 캔버스에 배치한다.
warped_img2 = cv.warpPerspective(img2_bgr, H, (panorama_width, panorama_height))

# 정렬 결과를 담을 최종 파노라마 이미지를 warped 결과로 초기화한다.
panorama_bgr = warped_img2.copy()

# 좌측 상단 기준으로 img1 원본을 파노라마에 올린다.
panorama_bgr[0:h1, 0:w1] = img1_bgr

# 좌측 상단 영역에서 warped와 원본의 겹침 영역을 계산한다.
overlap_mask = (
  np.sum(warped_img2[0:h1, 0:w1], axis=2) > 0
) & (
  np.sum(img1_bgr, axis=2) > 0
)

# 원본 기준 합성 영역을 복사본으로 준비한다.
blended_region = img1_bgr.copy()

# 겹침 부분은 두 영상을 평균 혼합해 경계 이질감을 줄인다.
blended_region[overlap_mask] = (
  0.5 * img1_bgr[overlap_mask] + 0.5 * warped_img2[0:h1, 0:w1][overlap_mask]
).astype(np.uint8)

# 블렌딩된 좌측 상단 영역을 최종 파노라마에 반영한다.
panorama_bgr[0:h1, 0:w1] = blended_region

# 좋은 매칭 중 상위 120개를 미리보기용으로 제한한다.
preview_matches = good_matches[:120]

# 미리보기용 매칭 결과 이미지를 생성한다.
match_preview_bgr = cv.drawMatches(
  img1_bgr,
  kp1,
  img2_bgr,
  kp2,
  preview_matches,
  None,
  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# RANSAC inlier 매칭 전체를 시각화한다.
match_inlier_bgr = cv.drawMatches(
  img1_bgr,
  kp1,
  img2_bgr,
  kp2,
  inlier_matches,
  None,
  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# 중간 결과(매칭 미리보기)를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "homography_matching_preview.jpg"), match_preview_bgr)

# 중간 결과(inlier 매칭)를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "homography_matching_inliers.jpg"), match_inlier_bgr)

# 중간 결과(warped 이미지)를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "homography_warped.jpg"), warped_img2)

# 최종 결과(정렬/합성 이미지)를 파일로 저장한다.
cv.imwrite(str(RESULT_DIR / "homography_aligned.jpg"), panorama_bgr)

# Matplotlib 표시를 위해 매칭 미리보기 이미지를 RGB로 변환한다.
match_preview_rgb = cv.cvtColor(match_preview_bgr, cv.COLOR_BGR2RGB)

# Matplotlib 표시를 위해 최종 파노라마 이미지를 RGB로 변환한다.
panorama_rgb = cv.cvtColor(panorama_bgr, cv.COLOR_BGR2RGB)

# 결과 비교용 Figure를 생성한다.
plt.figure(figsize=(16, 6))

# 좌측 패널에 매칭 결과를 배치한다.
plt.subplot(1, 2, 1)

# 좌측 패널에 매칭 이미지를 출력한다.
plt.imshow(match_preview_rgb)

# 좌측 패널 제목을 지정한다.
plt.title("Matching Result (Top 120 Good Matches)")

# 좌측 패널 축 눈금을 숨긴다.
plt.axis("off")

# 우측 패널에 정렬 결과를 배치한다.
plt.subplot(1, 2, 2)

# 우측 패널에 정렬 이미지를 출력한다.
plt.imshow(panorama_rgb)

# 우측 패널 제목을 지정한다.
plt.title("Warped + Aligned Result")

# 우측 패널 축 눈금을 숨긴다.
plt.axis("off")

# 전체 레이아웃이 겹치지 않도록 정리한다.
plt.tight_layout()

# 과제용 최종 시각화 이미지를 저장한다.
plt.savefig(str(RESULT_DIR / "homography_visualization.png"), dpi=150)

# 메모리 해제를 위해 Figure를 닫는다.
plt.close()

# 실험 요약 정보를 줄 단위 문자열로 구성한다.
summary_lines = [
  f"Image1: {IMAGE1_PATH}",
  f"Image2: {IMAGE2_PATH}",
  f"Keypoints image1: {len(kp1)}",
  f"Keypoints image2: {len(kp2)}",
  f"Good matches: {len(good_matches)}",
  f"Inlier matches (RANSAC): {len(inlier_matches)}",
  "Homography matrix:",
  np.array2string(H, precision=6, suppress_small=False),
]

# 요약 정보를 텍스트 파일로 저장한다.
(RESULT_DIR / "homography_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# 콘솔에 첫 번째 이미지 경로를 출력한다.
print(f"Image1: {IMAGE1_PATH}")

# 콘솔에 두 번째 이미지 경로를 출력한다.
print(f"Image2: {IMAGE2_PATH}")

# 콘솔에 첫 번째 이미지 특징점 수를 출력한다.
print(f"Keypoints image1: {len(kp1)}")

# 콘솔에 두 번째 이미지 특징점 수를 출력한다.
print(f"Keypoints image2: {len(kp2)}")

# 콘솔에 좋은 매칭 수를 출력한다.
print(f"Good matches: {len(good_matches)}")

# 콘솔에 RANSAC inlier 매칭 수를 출력한다.
print(f"Inlier matches (RANSAC): {len(inlier_matches)}")

# 콘솔에 호모그래피 행렬을 출력한다.
print("Homography matrix:")

# 콘솔에 호모그래피 행렬 값을 출력한다.
print(H)

# 콘솔에 결과 저장 위치를 출력한다.
print(f"Saved results to: {RESULT_DIR}")
```


# Week2

## 01.체크보드 기반 카메라 캘리브레이션

### 과제 설명
- 체크보드 이미지에서 코너를 검출하고, 실제 3D 좌표-이미지 2D 좌표 대응으로 카메라 내부 파라미터를 추정한다.
- 여러 장의 체크보드 이미지로 `cv2.calibrateCamera()`를 수행하여 카메라 내부 행렬 `K`와 왜곡 계수 `dist`를 계산한다.
- 계산된 파라미터를 사용해 `cv2.undistort()`로 왜곡 보정을 수행하고 시각화 결과를 저장한다.

### 요구사항 반영 체크
- 모든 이미지에서 체크보드 코너 검출 시도: 완료
- 실제 좌표(한 칸 25mm)와 이미지 코너 좌표 구성: 완료
- `cv2.calibrateCamera()`로 `K`, `dist` 계산: 완료
- `cv2.undistort()`로 왜곡 보정 시각화: 완료
- 코너 검출 실패 이미지 제외 처리: 완료

### 중간 결과물
- 코너 검출 시각화 이미지(13장)
- 저장 경로: `week2/results_calibration/`
- 파일 목록:
  - `corners_left01.jpg`
  - `corners_left02.jpg`
  - `corners_left03.jpg`
  - `corners_left04.jpg`
  - `corners_left05.jpg`
  - `corners_left06.jpg`
  - `corners_left07.jpg`
  - `corners_left08.jpg`
  - `corners_left09.jpg`
  - `corners_left10.jpg`
  - `corners_left11.jpg`
  - `corners_left12.jpg`
  - `corners_left13.jpg`

### 최종 결과물
- 왜곡 보정 전/후 비교 이미지:
  ![Image](https://github.com/user-attachments/assets/f2fe61e6-1210-4356-b401-a634b681e535)
- 캘리브레이션 수치 결과(npz):
  - `week2/results_calibration/calibration_result.npz`
- 콘솔 출력 결과:
```text
RMS Reprojection Error: 0.408695

Camera Matrix K:
[[536.07345314   0.         342.37046827]
 [  0.         536.01636274 235.53687064]
 [  0.           0.           1.        ]]

Distortion Coefficients:
[[-0.26509039 -0.0467422   0.00183302 -0.00031469  0.25231221]]

Total images: 13
Detected corners: 13
Failed detections: 0
```

### 코드 (주석 포함)
```python
import glob  # 파일 패턴 검색을 위한 표준 모듈
from pathlib import Path  # 운영체제와 무관한 경로 처리를 위한 모듈

import cv2  # OpenCV 라이브러리
import numpy as np  # 수치 연산용 라이브러리

# 체크보드의 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기(mm)
SQUARE_SIZE_MM = 25.0

# 코너 정밀화(sub-pixel) 반복 조건: 최대 30회 또는 오차 0.001 이하
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 현재 파이썬 파일이 있는 폴더 경로
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로
RESULT_DIR = SCRIPT_DIR / "results_calibration"

# 결과 폴더가 없으면 생성
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 3D 실제 좌표 템플릿 생성 (Z=0 평면)
objp_template = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# 2D 격자 좌표를 (x, y)로 채움
objp_template[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 격자 좌표를 실제 길이(mm) 단위로 스케일링
objp_template *= SQUARE_SIZE_MM

# 각 이미지에 대응되는 3D 실제 좌표 리스트
objpoints = []

# 각 이미지에서 검출한 2D 코너 좌표 리스트
imgpoints = []

# 캘리브레이션 이미지가 저장된 실제 폴더 패턴 경로
image_pattern = str(SCRIPT_DIR.parent.parent / "images" / "calibration_images" / "left*.jpg")

# 지정된 경로의 캘리브레이션 이미지를 정렬하여 로드
images = sorted(glob.glob(image_pattern))

# 이미지 크기(가로, 세로)를 저장할 변수
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

# 입력 이미지가 없으면 즉시 종료
if not images:
    raise FileNotFoundError(
        "캘리브레이션 이미지를 찾지 못했습니다. "
        "경로: C:/Projects/Computer_vision/images/calibration_images/left*.jpg"
    )

# 코너 검출 성공 횟수 카운터
success_count = 0

# 코너 검출 실패 횟수 카운터
fail_count = 0

# 모든 캘리브레이션 이미지를 순회
for image_path in images:
    # 컬러 이미지 로드
    img = cv2.imread(image_path)

    # 이미지 로드 실패 시 해당 파일은 건너뜀
    if img is None:
        fail_count += 1
        continue

    # 그레이스케일 변환 (코너 검출용)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 크기 저장 (가로, 세로)
    img_size = gray.shape[::-1]

    # 체크보드 코너 검출
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 찾은 경우에만 좌표를 누적
    if found:
        # sub-pixel 정밀화로 코너 정확도 향상
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        # 현재 이미지에 대한 3D 좌표 추가
        objpoints.append(objp_template.copy())

        # 현재 이미지에 대한 2D 코너 좌표 추가
        imgpoints.append(corners_refined)

        # 디버깅/과제 제출용 시각화 이미지를 복사
        vis = img.copy()

        # 검출된 코너를 원본 위에 그림
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_refined, found)

        # 중간 결과 저장 파일명 구성
        stem = Path(image_path).stem
        out_path = RESULT_DIR / f"corners_{stem}.jpg"

        # 코너 시각화 이미지 저장
        cv2.imwrite(str(out_path), vis)

        # 성공 카운트 증가
        success_count += 1
    else:
        # 코너 미검출 이미지는 캘리브레이션에서 제외
        fail_count += 1

# 유효 이미지가 1장도 없으면 캘리브레이션 불가
if not objpoints or not imgpoints or img_size is None:
    raise RuntimeError("코너 검출에 성공한 이미지가 없어 캘리브레이션을 진행할 수 없습니다.")

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

# OpenCV 캘리브레이션 수행
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None,
)

# 재투영 오차(RMS) 출력
print(f"RMS Reprojection Error: {ret:.6f}")

# 내부 파라미터 행렬 출력
print("\nCamera Matrix K:")
print(K)

# 왜곡 계수 출력
print("\nDistortion Coefficients:")
print(dist)

# 사용한 이미지 통계 출력
print(f"\nTotal images: {len(images)}")
print(f"Detected corners: {success_count}")
print(f"Failed detections: {fail_count}")

# 수치 결과를 npz 파일로 저장
np.savez(
    RESULT_DIR / "calibration_result.npz",
    rms=ret,
    K=K,
    dist=dist,
    rvecs=np.array(rvecs, dtype=object),
    tvecs=np.array(tvecs, dtype=object),
)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------

# 비교 시각화용 원본 이미지 1장을 다시 로드
sample_img = cv2.imread(images[0])

# 로드 실패 시 예외 발생
if sample_img is None:
    raise RuntimeError("왜곡 보정 시각화용 샘플 이미지를 불러오지 못했습니다.")

# 왜곡 보정 이미지 생성
undistorted = cv2.undistort(sample_img, K, dist)

# 원본/보정 이미지를 좌우로 붙여서 비교 이미지 생성
comparison = np.hstack([sample_img, undistorted])

# 비교 이미지 파일 저장
comparison_path = RESULT_DIR / "undistort_comparison.jpg"
cv2.imwrite(str(comparison_path), comparison)

# 저장 경로 안내
print(f"\nSaved intermediate and final results to: {RESULT_DIR}")
```


## 02.이미지 Rotation & Transformation

### 과제 설명
- 한 장의 이미지에 회전, 크기 조절, 평행이동을 순차적으로 적용한다.
- 이미지 중심을 기준으로 +30도 회전하고, 동시에 크기를 0.8배로 축소한다.
- 회전 및 크기 조절 결과를 x축으로 +80px, y축으로 -40px 평행이동한다.
- `cv2.getRotationMatrix2D()`로 회전 행렬을 만들고, 마지막 열 값을 조정하여 평행이동을 반영한다.
- `cv2.warpAffine()`를 사용해 중간 결과와 최종 결과를 생성하고 저장한다.

### 요구사항 반영 체크
- 이미지 중심 기준 +30도 회전: 완료
- 회전과 동시에 0.8배 크기 조절: 완료
- x축 +80px, y축 -40px 평행이동: 완료
- `cv2.getRotationMatrix2D()` 사용: 완료
- `cv2.warpAffine()` 사용: 완료
- 회전 행렬 마지막 열 조정으로 평행이동 반영: 완료

### 중간 결과물
- 회전 + 크기 조절 결과 이미지:
   ![Image](https://github.com/user-attachments/assets/a06d119e-ecd3-484f-a30b-8298884e0efd)
- 변환 행렬 정보 텍스트:
    - `Image size: 1188 x 792
      Center: (594.0, 396.0)

      Rotation matrix:
      [[  0.69282032   0.4         24.06472812]
       [ -0.4          0.69282032 359.24315208]]
      
      Transformation matrix:
      [[  0.69282032   0.4        104.06472812]
       [ -0.4          0.69282032 319.24315208]]
      `

### 최종 결과물
- 회전 + 크기 조절 + 평행이동 결과 이미지:
    ![Image](https://github.com/user-attachments/assets/fe4bb817-6756-4c40-9864-f403f1752752)
- 원본 / 중간 / 최종 비교 이미지:
    ![Image](https://github.com/user-attachments/assets/786462fd-6431-4f5d-985f-8a115b3f8b52)
- 콘솔 출력 결과:
```text
Input image: C:\Projects\Computer_vision\images\rose.png
Image size: 1188 x 792

Rotation matrix:
[[  0.69282032   0.4         24.06472812]
 [ -0.4          0.69282032 359.24315208]]

Transformation matrix:
[[  0.69282032   0.4        104.06472812]
 [ -0.4          0.69282032 319.24315208]]
```

### 코드 (주석 포함)
```python
from pathlib import Path  # 운영체제에 독립적인 경로 처리를 위한 모듈

import cv2  # OpenCV 라이브러리
import numpy as np  # 수치 계산 및 행렬 출력을 위한 라이브러리

# 회전 각도 설정(+30도)
ANGLE_DEG = 30

# 회전과 동시에 적용할 크기 비율(0.8배)
SCALE = 0.8

# x축 방향 평행이동 값(+80픽셀)
TRANSLATE_X = 80

# y축 방향 평행이동 값(-40픽셀)
TRANSLATE_Y = -40

# 현재 파이썬 파일이 있는 폴더 경로
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로
RESULT_DIR = SCRIPT_DIR / "results_rotation"

# 결과 폴더가 없으면 생성
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 회전 변환에 사용할 원본 이미지의 실제 경로
image_path = SCRIPT_DIR.parent.parent / "images" / "rose.png"

# 지정한 입력 이미지가 없으면 예외 발생
if not image_path.exists():
    raise FileNotFoundError(f"회전/변환에 사용할 입력 이미지를 찾지 못했습니다: {image_path}")

# 원본 이미지 로드
image = cv2.imread(str(image_path))

# 이미지 로드 실패 시 예외 발생
if image is None:
        raise RuntimeError(f"입력 이미지 로드에 실패했습니다: {image_path}")

# 원본 이미지의 높이와 너비 추출
height, width = image.shape[:2]

# 이미지 중심 좌표 계산
center = (width / 2, height / 2)

# 회전 + 스케일 변환 행렬 생성
rotation_matrix = cv2.getRotationMatrix2D(center, ANGLE_DEG, SCALE)

# 중간 결과용: 회전과 스케일만 적용한 이미지 생성
rotated_scaled = cv2.warpAffine(image, rotation_matrix, (width, height))

# 최종 결과용: 회전 행렬을 복사하여 평행이동까지 반영
transformation_matrix = rotation_matrix.copy()

# x축 평행이동 값을 마지막 열에 더함
transformation_matrix[0, 2] += TRANSLATE_X

# y축 평행이동 값을 마지막 열에 더함
transformation_matrix[1, 2] += TRANSLATE_Y

# 회전 + 스케일 + 평행이동을 한 번에 적용한 최종 이미지 생성
transformed = cv2.warpAffine(image, transformation_matrix, (width, height))

# 원본, 중간 결과, 최종 결과를 가로로 이어 붙임
comparison = np.hstack([image, rotated_scaled, transformed])

# 중간 결과 이미지 저장
cv2.imwrite(str(RESULT_DIR / "rotated_scaled.jpg"), rotated_scaled)

# 최종 결과 이미지 저장
cv2.imwrite(str(RESULT_DIR / "rotated_scaled_translated.jpg"), transformed)

# 비교 이미지 저장
cv2.imwrite(str(RESULT_DIR / "rotation_comparison.jpg"), comparison)

# 변환 행렬 정보를 텍스트 파일로 저장
matrix_text = (
        f"Input image: {image_path}\n"
        f"Image size: {width} x {height}\n"
        f"Center: {center}\n\n"
        f"Rotation matrix:\n{rotation_matrix}\n\n"
        f"Transformation matrix:\n{transformation_matrix}\n"
)

# 행렬 정보 파일 저장
(RESULT_DIR / "transformation_matrices.txt").write_text(matrix_text, encoding="utf-8")

# 사용한 입력 이미지 출력
print(f"Input image: {image_path}")

# 원본 이미지 크기 출력
print(f"Image size: {width} x {height}")

# 회전 + 스케일 행렬 출력
print("\nRotation matrix:")
print(rotation_matrix)

# 회전 + 스케일 + 평행이동 행렬 출력
print("\nTransformation matrix:")
print(transformation_matrix)

# 결과 저장 폴더 출력
print(f"\nSaved results to: {RESULT_DIR}")
```


## 03.Stereo Disparity 기반 Depth 추정

### 과제 설명
- 같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 사용해 깊이를 추정한다.
- 두 이미지에서 같은 물체가 얼마나 수평으로 이동했는지 disparity를 계산하고, 이를 이용해 depth를 구한다.
- `cv2.StereoBM_create()`로 disparity map을 계산하고, `Z = fB / d` 공식을 이용해 depth map을 계산한다.
- ROI `Painting`, `Frog`, `Teddy` 각각에 대해 평균 disparity와 평균 depth를 계산하고, 가장 가까운 영역과 가장 먼 영역을 해석한다.

### 요구사항 반영 체크
- 입력 이미지를 그레이스케일로 변환: 완료
- `cv2.StereoBM_create()`로 disparity map 계산: 완료
- `Disparity > 0` 인 픽셀만 사용해 depth map 계산: 완료
- ROI `Painting`, `Frog`, `Teddy`의 평균 disparity / depth 계산: 완료
- 가장 가까운 ROI / 가장 먼 ROI 해석: 완료
- disparity / depth 시각화 결과 저장: 완료

### 중간 결과물
- ROI가 표시된 좌측 이미지:

  <img width="450" height="375" alt="Image" src="https://github.com/user-attachments/assets/3ff02b62-9f03-432b-bf3a-df37efd8aa8d" />
- ROI가 표시된 우측 이미지:

  <img width="450" height="375" alt="Image" src="https://github.com/user-attachments/assets/e6e080c9-73eb-4f49-b961-343b90f9fa8e" />
- 컬러 disparity map:

  <img width="450" height="375" alt="Image" src="https://github.com/user-attachments/assets/6627cc70-67e5-463e-95d4-db68fec76e06" />
- ROI별 수치 결과 텍스트:
  - `Focal length (f): 700.0
    Baseline (B): 0.12
    
    ROI Statistics:
    Painting: mean disparity = 18.5413, mean depth = 4.5373 m, valid pixels = 9130
    Frog: mean disparity = 33.6996, mean depth = 2.5056 m, valid pixels = 18115
    Teddy: mean disparity = 22.4374, mean depth = 3.8561 m, valid pixels = 9170
    
    Nearest ROI: Frog
    Farthest ROI: Painting`

### 최종 결과물
- 컬러 depth map:

  <img width="450" height="375" alt="Image" src="https://github.com/user-attachments/assets/b8b0d809-3170-4352-b5a3-f8fe24dc1807" />
- 전체 요약 패널(좌측/우측/Disparity/Depth):

  <img width="900" height="750" alt="Image" src="https://github.com/user-attachments/assets/acea50eb-811f-4c2b-8e64-5a3e6e8a9cf5" />
- 콘솔 출력 결과:
```text
Left image: C:\Projects\Computer_vision\images\left.png
Right image: C:\Projects\Computer_vision\images\right.png

Focal length (f): 700.0
Baseline (B): 0.12

ROI Statistics:
- Painting: mean disparity = 18.5413, mean depth = 4.5373 m, valid pixels = 9130
- Frog: mean disparity = 33.6996, mean depth = 2.5056 m, valid pixels = 18115
- Teddy: mean disparity = 22.4374, mean depth = 3.8561 m, valid pixels = 9170

Nearest ROI: Frog
Farthest ROI: Painting
```

### 결과 해석
- `Frog` 영역의 평균 disparity가 가장 크고 평균 depth가 가장 작으므로 카메라에 가장 가깝다.
- `Painting` 영역의 평균 disparity가 가장 작고 평균 depth가 가장 크므로 카메라에서 가장 멀다.
- `Teddy` 영역은 두 값이 중간 수준이므로 거리도 중간 정도로 해석할 수 있다.

### 코드 (주석 포함)
```python
from pathlib import Path  # 운영체제에 독립적인 경로 처리를 위한 모듈

import cv2  # OpenCV 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리

# 현재 파이썬 파일이 있는 폴더 경로
SCRIPT_DIR = Path(__file__).resolve().parent

# 결과 이미지를 저장할 폴더 경로
RESULT_DIR = SCRIPT_DIR / "results_depth"

# 결과 폴더가 없으면 생성
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 좌측 스테레오 이미지의 실제 경로
LEFT_IMAGE_PATH = SCRIPT_DIR.parent.parent / "images" / "left.png"

# 우측 스테레오 이미지의 실제 경로
RIGHT_IMAGE_PATH = SCRIPT_DIR.parent.parent / "images" / "right.png"

# 카메라 초점 거리(focal length)
FOCAL_LENGTH = 700.0

# 카메라 간 거리(baseline, meter)
BASELINE = 0.12

# ROI 이름과 위치 정보(x, y, w, h)
ROIS = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90),
}

# 좌측 이미지가 존재하지 않으면 예외 발생
if not LEFT_IMAGE_PATH.exists():
    raise FileNotFoundError(f"좌측 이미지를 찾지 못했습니다: {LEFT_IMAGE_PATH}")

# 우측 이미지가 존재하지 않으면 예외 발생
if not RIGHT_IMAGE_PATH.exists():
    raise FileNotFoundError(f"우측 이미지를 찾지 못했습니다: {RIGHT_IMAGE_PATH}")

# 좌측 컬러 이미지 로드
left_color = cv2.imread(str(LEFT_IMAGE_PATH))

# 우측 컬러 이미지 로드
right_color = cv2.imread(str(RIGHT_IMAGE_PATH))

# 이미지 로드 실패 시 예외 발생
if left_color is None or right_color is None:
    raise RuntimeError("좌/우 이미지를 불러오는 데 실패했습니다.")

# 좌측 이미지를 그레이스케일로 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)

# 우측 이미지를 그레이스케일로 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# StereoBM 객체 생성 (numDisparities는 16의 배수여야 함)
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)

# 밝기 차이를 줄이기 위해 히스토그램 평활화 수행
left_gray_eq = cv2.equalizeHist(left_gray)

# 우측 이미지에도 동일하게 히스토그램 평활화 수행
right_gray_eq = cv2.equalizeHist(right_gray)

# -----------------------------
# 1. Disparity 계산
# -----------------------------

# StereoBM 결과를 float32로 변환하고 16으로 나눠 실제 disparity 값으로 복원
disparity = stereo.compute(left_gray_eq, right_gray_eq).astype(np.float32) / 16.0

# disparity가 0보다 큰 영역만 유효한 영역으로 정의
valid_mask = disparity > 0

# 유효한 disparity가 하나도 없으면 예외 발생
if not np.any(valid_mask):
    raise ValueError("유효한 disparity 값이 없습니다.")

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------

# depth 결과를 저장할 배열을 0으로 초기화
depth_map = np.zeros_like(disparity, dtype=np.float32)

# disparity가 유효한 위치에만 깊이 값을 계산
depth_map[valid_mask] = (FOCAL_LENGTH * BASELINE) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------

# ROI별 통계 결과를 저장할 딕셔너리
results = {}

# 모든 ROI를 순회하며 통계를 계산
for name, (x, y, w, h) in ROIS.items():
    # 현재 ROI의 disparity 영역 추출
    roi_disparity = disparity[y:y + h, x:x + w]

    # 현재 ROI의 depth 영역 추출
    roi_depth = depth_map[y:y + h, x:x + w]

    # 현재 ROI에서 유효한 disparity 마스크 추출
    roi_valid = roi_disparity > 0

    # ROI 내부에 유효한 픽셀이 있으면 평균값 계산
    if np.any(roi_valid):
        # 유효한 disparity 평균 계산
        mean_disparity = float(np.mean(roi_disparity[roi_valid]))

        # 유효한 depth 평균 계산
        mean_depth = float(np.mean(roi_depth[roi_valid]))

        # 유효 픽셀 개수 계산
        valid_pixels = int(np.count_nonzero(roi_valid))
    else:
        # 유효 픽셀이 없으면 NaN 처리
        mean_disparity = float("nan")

        # 유효 픽셀이 없으면 NaN 처리
        mean_depth = float("nan")

        # 유효 픽셀 수는 0으로 기록
        valid_pixels = 0

    # 결과 딕셔너리에 현재 ROI 통계 저장
    results[name] = {
        "mean_disparity": mean_disparity,
        "mean_depth": mean_depth,
        "valid_pixels": valid_pixels,
    }

# 유효한 depth 값을 가진 ROI 이름만 추출
valid_roi_names = [name for name, stats in results.items() if not np.isnan(stats["mean_depth"])]

# 유효한 ROI가 하나도 없으면 예외 발생
if not valid_roi_names:
    raise ValueError("ROI 내부에서 유효한 disparity/depth 값을 찾지 못했습니다.")

# 평균 depth가 가장 작은 ROI를 가장 가까운 영역으로 결정
nearest_roi = min(valid_roi_names, key=lambda name: results[name]["mean_depth"])

# 평균 depth가 가장 큰 ROI를 가장 먼 영역으로 결정
farthest_roi = max(valid_roi_names, key=lambda name: results[name]["mean_depth"])

# -----------------------------
# 4. 결과 출력
# -----------------------------

# 입력 이미지 경로 출력
print(f"Left image: {LEFT_IMAGE_PATH}")

# 우측 이미지 경로 출력
print(f"Right image: {RIGHT_IMAGE_PATH}")

# 카메라 파라미터 출력
print(f"\nFocal length (f): {FOCAL_LENGTH}")
print(f"Baseline (B): {BASELINE}")

# 각 ROI 결과를 순서대로 출력
print("\nROI Statistics:")
for name, stats in results.items():
    print(
        f"- {name}: mean disparity = {stats['mean_disparity']:.4f}, "
        f"mean depth = {stats['mean_depth']:.4f} m, "
        f"valid pixels = {stats['valid_pixels']}"
    )

# 가장 가까운 ROI 출력
print(f"\nNearest ROI: {nearest_roi}")

# 가장 먼 ROI 출력
print(f"Farthest ROI: {farthest_roi}")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# 시각화용 disparity 복사본 생성
disp_tmp = disparity.copy()

# 유효하지 않은 disparity는 NaN으로 설정
disp_tmp[disp_tmp <= 0] = np.nan

# 유효 disparity의 하위 5% 값을 최소값으로 사용
d_min = np.nanpercentile(disp_tmp, 5)

# 유효 disparity의 상위 95% 값을 최대값으로 사용
d_max = np.nanpercentile(disp_tmp, 95)

# 최대값과 최소값이 같으면 분모 보호
if d_max <= d_min:
    d_max = d_min + 1e-6

# disparity를 0~1 구간으로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)

# 0~1 범위를 벗어나지 않도록 제한
disp_scaled = np.clip(disp_scaled, 0, 1)

# 시각화용 8비트 disparity 이미지 생성
disp_vis = np.zeros_like(disparity, dtype=np.uint8)

# 유효 disparity 위치 마스크 생성
valid_disp = ~np.isnan(disp_tmp)

# 유효 disparity만 0~255 범위로 변환
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# JET 컬러맵 적용 (빨강이 더 가까움)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------

# 시각화용 8비트 depth 이미지 생성
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

# 유효 depth 픽셀이 있으면 정규화 수행
if np.any(valid_mask):
    # 유효한 depth 값만 추출
    depth_valid = depth_map[valid_mask]

    # 하위 5% 깊이 값을 최소값으로 사용
    z_min = np.percentile(depth_valid, 5)

    # 상위 95% 깊이 값을 최대값으로 사용
    z_max = np.percentile(depth_valid, 95)

    # 최대값과 최소값이 같으면 분모 보호
    if z_max <= z_min:
        z_max = z_min + 1e-6

    # depth를 0~1 범위로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)

    # 0~1 범위를 벗어나지 않도록 제한
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 값이 작을수록 가까우므로 반전
    depth_scaled = 1.0 - depth_scaled

    # 유효한 위치만 0~255 범위로 변환
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# JET 컬러맵 적용
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------

# 좌측 이미지에 ROI를 그릴 복사본 생성
left_vis = left_color.copy()

# 우측 이미지에 ROI를 그릴 복사본 생성
right_vis = right_color.copy()

# 모든 ROI를 좌/우 이미지에 그림
for name, (x, y, w, h) in ROIS.items():
    # 좌측 이미지에 ROI 사각형 그림
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 좌측 이미지에 ROI 이름 표시
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 우측 이미지에 ROI 사각형 그림
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 우측 이미지에 ROI 이름 표시
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# disparity 컬러맵 위에 ROI를 그림
for name, (x, y, w, h) in ROIS.items():
    # disparity 이미지에 ROI 사각형 그림
    cv2.rectangle(disparity_color, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # disparity 이미지에 ROI 이름 표시
    cv2.putText(disparity_color, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # depth 이미지에 ROI 사각형 그림
    cv2.rectangle(depth_color, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # depth 이미지에 ROI 이름 표시
    cv2.putText(depth_color, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# -----------------------------
# 8. 저장
# -----------------------------

# 좌측 ROI 표시 이미지 저장
cv2.imwrite(str(RESULT_DIR / "left_with_rois.png"), left_vis)

# 우측 ROI 표시 이미지 저장
cv2.imwrite(str(RESULT_DIR / "right_with_rois.png"), right_vis)

# disparity 컬러맵 저장
cv2.imwrite(str(RESULT_DIR / "disparity_map_color.png"), disparity_color)

# depth 컬러맵 저장
cv2.imwrite(str(RESULT_DIR / "depth_map_color.png"), depth_color)

# 좌측 이미지와 disparity 비교 이미지를 생성
comparison_top = np.hstack([left_vis, disparity_color])

# 우측 이미지와 depth 비교 이미지를 생성
comparison_bottom = np.hstack([right_vis, depth_color])

# 전체 결과 패널 이미지 생성
summary_panel = np.vstack([comparison_top, comparison_bottom])

# 전체 결과 패널 저장
cv2.imwrite(str(RESULT_DIR / "depth_summary_panel.png"), summary_panel)

# ROI 결과 텍스트를 저장할 문자열 목록 생성
result_lines = [
    f"Left image: {LEFT_IMAGE_PATH}",
    f"Right image: {RIGHT_IMAGE_PATH}",
    f"Focal length (f): {FOCAL_LENGTH}",
    f"Baseline (B): {BASELINE}",
    "",
    "ROI Statistics:",
]

# 각 ROI의 통계를 텍스트 줄로 추가
for name, stats in results.items():
    result_lines.append(
        f"{name}: mean disparity = {stats['mean_disparity']:.4f}, "
        f"mean depth = {stats['mean_depth']:.4f} m, "
        f"valid pixels = {stats['valid_pixels']}"
    )

# 가장 가까운 ROI 정보를 텍스트에 추가
result_lines.append("")
result_lines.append(f"Nearest ROI: {nearest_roi}")

# 가장 먼 ROI 정보를 텍스트에 추가
result_lines.append(f"Farthest ROI: {farthest_roi}")

# 텍스트 결과 파일 저장
(RESULT_DIR / "roi_depth_results.txt").write_text("\n".join(result_lines), encoding="utf-8")

# -----------------------------
# 9. 출력
# -----------------------------

# 결과 저장 폴더 출력
print(f"\nSaved results to: {RESULT_DIR}")
```

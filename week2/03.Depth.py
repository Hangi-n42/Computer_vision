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

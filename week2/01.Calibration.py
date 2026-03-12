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


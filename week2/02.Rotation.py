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
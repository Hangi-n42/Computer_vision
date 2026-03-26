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

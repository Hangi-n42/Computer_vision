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

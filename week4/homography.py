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

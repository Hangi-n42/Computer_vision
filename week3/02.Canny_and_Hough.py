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

# Week6

## 01. SORT 알고리즘을 활용한 다중 객체 추적기 구현

### 과제 설명
- 본 과제는 비디오에서 여러 객체를 동시에 추적하기 위해 SORT(Simple Online and Realtime Tracking) 파이프라인을 구현하는 것을 목표로 한다.
- 객체 검출 단계는 OpenCV DNN 모듈과 YOLOv3를 사용하여 수행한다.
- 추적 단계는 칼만 필터로 객체 상태를 예측하고, IoU 기반 비용 행렬에 Hungarian 알고리즘을 적용하여 검출 결과와 기존 트랙을 연관한다.
- 각 트랙에는 고유 ID를 부여하고, 프레임 위에 ID/클래스/신뢰도를 표시해 실시간 시각화를 제공한다.
- 검출기는 CUDA 백엔드와 CUDA 타깃으로 설정하여 GPU를 사용하도록 구성한다.
- 입력 비디오는 `images/slow_traffic_small.mp4`, YOLO 설정/가중치는 `images/yolov3.cfg`, `images/yolov3.weights`를 사용한다.

### 중간 결과물 (설명 포함)
- `week6/results_sort/sort_mid_detection_frame.jpg`
  - 프레임 30에서 저장되는 중간 결과 이미지이다.
  - YOLOv3의 원시 검출 결과(DET 라벨, 클래스명, confidence, bbox)를 확인하여 검출 품질을 점검할 수 있다.
- `week6/results_sort/sort_mid_tracking_frame.jpg`
  - 프레임 60에서 저장되는 중간 결과 이미지이다.
  - SORT 연관 이후 부여된 ID와 함께 트랙이 안정적으로 유지되는지 확인할 수 있다.
- `week6/results_sort/sort_summary.txt`
  - 처리 프레임 수, 생성된 고유 ID 수, 평균 처리 FPS, CUDA 백엔드/타깃 사용 정보가 기록된다.

### 최종 결과물 (설명 포함)
- `week6/results_sort/sort_tracking_output.mp4`
  - 전체 비디오 구간에서 다중 객체 추적이 적용된 최종 결과 영상이다.
  - 각 객체는 `ID | class | confidence` 형식으로 표시되어 프레임 간 동일 객체 추적 여부를 확인할 수 있다.
- `week6/results_sort/sort_final_frame.jpg`
  - 추적 파이프라인의 마지막 처리 프레임 스냅샷이다.
  - 최종 시점에서의 추적 상태와 오버레이 정보를 정적으로 확인할 수 있다.

### 코드 (주석 포함)
```python
﻿from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
RESULT_DIR = SCRIPT_DIR / "results_sort"
VIDEO_PATH = IMAGES_DIR / "slow_traffic_small.mp4"
YOLO_CFG_PATH = IMAGES_DIR / "yolov3.cfg"
YOLO_WEIGHTS_PATH = IMAGES_DIR / "yolov3.weights"
OUTPUT_VIDEO_PATH = RESULT_DIR / "sort_tracking_output.mp4"
OUTPUT_SUMMARY_PATH = RESULT_DIR / "sort_summary.txt"
OUTPUT_MID_DETECTION_PATH = RESULT_DIR / "sort_mid_detection_frame.jpg"
OUTPUT_MID_TRACKING_PATH = RESULT_DIR / "sort_mid_tracking_frame.jpg"
OUTPUT_FINAL_FRAME_PATH = RESULT_DIR / "sort_final_frame.jpg"

# 교통 장면에서 사용되는 사람/자동차/오토바이/버스/트럭의 COCO 클래스 ID
TARGET_CLASS_IDS = {0, 2, 3, 5, 7}

# 클래스 이름은 각 추적 객체의 간결한 레이블 렌더링에만 사용됨
COCO_NAME_MAP: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    # 겹치는 영역의 좌상단 모서리 계산
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))

    # 겹치는 영역의 우하단 모서리 계산
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    # 박스가 겹치지 않으면 IoU는 정확히 0
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # 교집합 영역 계산
    intersection = (x_right - x_left) * (y_bottom - y_top)

    # 각 박스의 면적 계산
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))

    # 합집합 면적 계산 및 0으로 나누기 방지
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    # 최종 IoU 비율 반환
    return intersection / union


def hungarian_algorithm(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    # 불필요한 처리를 피하기 위해 빈 행렬 조기 처리
    if cost_matrix.size == 0:
        return []

    # 행렬 크기 읽기
    n_rows, n_cols = cost_matrix.shape

    # Hungarian 알고리즘은 정사각형 행렬을 사용하므로 필요시 패딩 추가
    n = max(n_rows, n_cols)

    # 패딩된 셀에는 기존 모든 비용보다 큰 값 사용
    pad_value = float(np.max(cost_matrix) + 1.0)

    # 패딩된 정사각형 비용 행렬 생성
    padded = np.full((n, n), pad_value, dtype=np.float32)

    # 원본 비용을 좌상단 블록에 복사
    padded[:n_rows, :n_cols] = cost_matrix

    # 행과 열의 듀얼 변수
    u = np.zeros(n + 1, dtype=np.float32)
    v = np.zeros(n + 1, dtype=np.float32)

    # 매칭 배열 (p[j] = 열 j에 매칭된 행)
    p = np.zeros(n + 1, dtype=np.int32)

    # 보강 경로 재구성에 사용되는 역추적 배열
    way = np.zeros(n + 1, dtype=np.int32)

    # 각 행을 반복하며 매칭 증강
    for i in range(1, n + 1):
        # 행 i에서 새로운 보강 경로 시작
        p[0] = i

        # 교대 경로의 현재 열을 0부터 시작
        j0 = 0

        # 축약 비용 최소값 및 방문 플래그 초기화
        minv = np.full(n + 1, np.inf, dtype=np.float32)
        used = np.zeros(n + 1, dtype=bool)

        # 자유 열을 찾을 때까지 보강 경로 확장
        while True:
            # 현재 열을 사용됨으로 표시
            used[j0] = True

            # 열 j0에 연결된 현재 행 검색
            i0 = p[j0]

            # 포텐셜 업데이트를 위해 최소 slack 추적
            delta = np.inf
            j1 = 0

            # 사용되지 않은 모든 열 이완
            for j in range(1, n + 1):
                # 이미 트리에 있는 열 건너뛰기
                if used[j]:
                    continue

                # 축약 비용 계산
                cur = padded[i0 - 1, j - 1] - u[i0] - v[j]

                # 열 j의 최적 선행자 업데이트
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0

                # 사용되지 않은 열 중 전역 최소 slack 추적
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            # 방문한 및 방문하지 않은 열의 포텐셜 업데이트
            for j in range(0, n + 1):
                # 교대 트리의 포텐셜 이동
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            # 최소 slack을 가진 다음 열로 이동
            j0 = j1

            # 매칭되지 않은 열에 도달하면 중단
            if p[j0] == 0:
                break

        # 보강 경로 재구성 및 매칭 업데이트
        while True:
            # 선행자 열 따라가기
            j1 = way[j0]

            # 열 j0을 j1에 이전에 있던 행에 재할당
            p[j0] = p[j1]

            # 역추적 계속
            j0 = j1

            # 루트 열에 도달하면 종료
            if j0 == 0:
                break

    # 할당 수집 및 패딩된 항목 제거
    assignment: List[Tuple[int, int]] = []

    # 열 기반 매칭을 (행, 열) 쌍으로 변환
    for j in range(1, n + 1):
        # p[j]는 1부터 시작하는 행 인덱스
        row = int(p[j]) - 1

        # j는 1부터 시작하는 열 인덱스
        col = j - 1

        # 원본 행렬 범위 내의 쌍만 유지
        if 0 <= row < n_rows and 0 <= col < n_cols:
            assignment.append((row, col))

    # 최적 할당 쌍 반환
    return assignment


class KalmanTrack:
    # 안정적인 객체 ID 생성을 위한 공유 추적 ID 카운터
    next_id = 1

    def __init__(self, detection: np.ndarray):
        # 이 추적의 고유 정수 식별자 저장
        self.track_id = KalmanTrack.next_id

        # 다음에 생성될 추적을 위해 공유 카운터 증가
        KalmanTrack.next_id += 1

        # 8차원 정속도 모델 칼만 필터: 박스 모서리 [x1,y1,x2,y2]와 속도 [vx1,vy1,vx2,vy2]를 상태로 추적하여 객체 움직임을 예측
        self.kf = cv.KalmanFilter(8, 4)

        # 상태 전이 행렬: 각 모서리 좌표에 대해 위치 += 속도로 모델링
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # 측정 행렬: 상태를 관측된 상자 모서리 [x1,y1,x2,y2]로 투영
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # 프로세스 노이즈: 추적이 움직임 변화에 얼마나 빠르게 적응할지 제어
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2

        # 측정 노이즈: 검출기 측정값을 얼마나 신뢰할지 제어
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2

        # 사후 오차 공분산: 상태 추정의 초기 불확실성 설정
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # 첫 번째 검출값으로 위치를 초기화하고 속도는 0으로 설정
        x1, y1, x2, y2 = detection[:4].astype(np.float32)
        self.kf.statePost = np.array([[x1], [y1], [x2], [y2], [0], [0], [0], [0]], dtype=np.float32)

        # 트랙 메타데이터: SORT 생명주기 카운터 유지
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # 렌더링을 위해 최신 시각 속성 유지
        self.confidence = float(detection[4])
        self.class_id = int(detection[5])

    def predict(self) -> np.ndarray:
        # 한 타임스텝 앞의 상태를 예측
        prediction = self.kf.predict()

        # 매칭 여부와 무관하게 프레임마다 트랙 나이 증가
        self.age += 1

        # 다음 업데이트가 오기 전까지 미스 프레임 카운터 증가
        self.time_since_update += 1

        # 현재 프레임이 미매칭이면 연속 히트 streak를 리셋
        if self.time_since_update > 0:
            self.hit_streak = 0

        # 예측된 상태 벡터를 유효한 [x1,y1,x2,y2] 박스로 변환
        return self._state_to_bbox(prediction)

    def update(self, detection: np.ndarray) -> None:
        # 검출기 출력으로부터 측정 벡터 구성
        measurement = detection[:4].astype(np.float32).reshape(4, 1)

        # 현재 관측값으로 칼만 상태 보정
        self.kf.correct(measurement)

        # 이 프레임에서 트랙이 업데이트되었음을 표시
        self.time_since_update = 0

        # 총 성공 연관 횟수 증가
        self.hits += 1

        # 연속 연관 streak 증가
        self.hit_streak += 1

        # 표시용으로 최신 검출 신뢰도와 클래스를 저장
        self.confidence = float(detection[4])
        self.class_id = int(detection[5])

    def get_state(self) -> np.ndarray:
        # 현재 사후 추정값을 정돈된 바운딩 박스로 반환
        return self._state_to_bbox(self.kf.statePost)

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        # 칼만 상태 벡터에서 모서리 좌표 읽기
        x1, y1, x2, y2 = state[:4].reshape(-1)

        # 수치 오차가 있어도 모서리 순서가 올바르도록 강제
        x1_fixed = min(float(x1), float(x2))
        y1_fixed = min(float(y1), float(y2))
        x2_fixed = max(float(x1), float(x2))
        y2_fixed = max(float(y1), float(y2))

        # 후속 계산 일관성을 위해 float32로 반환
        return np.array([x1_fixed, y1_fixed, x2_fixed, y2_fixed], dtype=np.float32)


class SortTracker:
    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        # 트랙 삭제 전 허용할 최대 미매칭 프레임 수
        self.max_age = max_age

        # 트랙을 확정 상태로 보기 위한 최소 총 히트 수
        self.min_hits = min_hits

        # 검출-트랙 연관을 수락할 IoU 임계값
        self.iou_threshold = iou_threshold

        # 활성 트랙을 저장하는 내부 리스트
        self.tracks: List[KalmanTrack] = []

        # min_hits 이전 초기 프레임에서 트랙 표시를 돕는 프레임 카운터
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> List[KalmanTrack]:
        # 생명주기 로직을 위해 처리 프레임 수 증가
        self.frame_count += 1

        # 현재 프레임의 모든 트랙 위치 예측
        predicted_boxes = []
        for track in self.tracks:
            predicted_boxes.append(track.predict())

        # 벡터화 연산을 위해 리스트를 numpy 배열로 변환
        if len(predicted_boxes) > 0:
            predicted_array = np.vstack(predicted_boxes)
        else:
            predicted_array = np.empty((0, 4), dtype=np.float32)

        # IoU + Hungarian 매칭으로 검출과 예측을 최적으로 연관
        matched, unmatched_tracks, unmatched_detections = self._associate(detections, predicted_array)

        # 매칭된 검출값으로 기존 트랙 업데이트
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])

        # 미매칭 검출값으로 새 트랙 생성
        for det_idx in unmatched_detections:
            self.tracks.append(KalmanTrack(detections[det_idx]))

        # max_age를 넘지 않은 트랙만 남기고 오래된 트랙 제거
        survivors: List[KalmanTrack] = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                survivors.append(track)
        self.tracks = survivors

        # 렌더링 대상 트랙 선택: 확정 트랙 또는 초기 부트스트랩 구간
        visible_tracks: List[KalmanTrack] = []
        for track in self.tracks:
            recently_updated = track.time_since_update == 0
            confirmed = track.hits >= self.min_hits
            warmup = self.frame_count <= self.min_hits
            if recently_updated and (confirmed or warmup):
                visible_tracks.append(track)

        # 화면에 표시할 활성 트랙 반환
        return visible_tracks

    def _associate(self, detections: np.ndarray, predicted_boxes: np.ndarray):
        # 연관 문제의 행렬 차원 읽기
        num_tracks = predicted_boxes.shape[0]
        num_detections = detections.shape[0]

        # 기존 트랙이 없는 경계 상황 처리
        if num_tracks == 0:
            unmatched_detections = list(range(num_detections))
            return [], [], unmatched_detections

        # 검출 결과가 없는 경계 상황 처리
        if num_detections == 0:
            unmatched_tracks = list(range(num_tracks))
            return [], unmatched_tracks, []

        # 각 트랙 예측과 검출 사이의 IoU 행렬 구성
        iou_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        for track_idx in range(num_tracks):
            for det_idx in range(num_detections):
                iou_matrix[track_idx, det_idx] = iou_xyxy(predicted_boxes[track_idx], detections[det_idx, :4])

        # IoU 최대화 문제를 비용 최소화 문제로 변환
        cost_matrix = 1.0 - iou_matrix

        # Hungarian 알고리즘으로 최적의 1:1 할당 계산
        assignment = hungarian_algorithm(cost_matrix)

        # 매칭/미매칭 결과를 담을 컨테이너 준비
        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(num_tracks))
        unmatched_detections = set(range(num_detections))

        # IoU 임계값으로 약한 매칭을 필터링해 제거
        for track_idx, det_idx in assignment:
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                matched.append((track_idx, det_idx))
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(det_idx)

        # 연관 결과를 리스트 형태로 반환
        return matched, sorted(unmatched_tracks), sorted(unmatched_detections)


class YoloV3Detector:
    def __init__(self, cfg_path: Path, weights_path: Path, conf_threshold: float = 0.35, nms_threshold: float = 0.45):
        # 필터링과 NMS에 사용할 검출 임계값 저장
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 검출기가 실제 CUDA로 동작 중인지 CPU 폴백인지 추적
        self.using_cuda = False

        # 요약 보고용 백엔드/타깃 문자열 보관
        self.backend_name = "cv.dnn.DNN_BACKEND_OPENCV"
        self.target_name = "cv.dnn.DNN_TARGET_CPU"

        # cfg와 weights 파일로 Darknet 네트워크 로드
        self.net = cv.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))

        # OpenCV에서 CUDA 디바이스가 보이면 GPU 경로 사용 시도
        if hasattr(cv, "cuda") and cv.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                # 추론 가속을 위해 CUDA 백엔드 요청
                self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

                # 네트워크가 GPU에서 실행되도록 CUDA 타깃 요청
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

                # 설정이 성공한 경우에만 CUDA 활성 상태로 표시
                self.using_cuda = True
                self.backend_name = "cv.dnn.DNN_BACKEND_CUDA"
                self.target_name = "cv.dnn.DNN_TARGET_CUDA"
            except cv.error:
                # 현재 OpenCV 빌드가 CUDA DNN을 못 쓰면 CPU로 폴백
                self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        else:
            # CUDA 장치가 없으면 CPU 실행 유지
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # 반복 조회를 줄이기 위해 출력 레이어 이름을 캐시
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame: np.ndarray) -> np.ndarray:
        # 정규화된 YOLO 출력을 픽셀 좌표로 바꾸기 위해 프레임 크기 읽기
        height, width = frame.shape[:2]

        # 표준 YOLOv3 전처리로 입력 blob 생성
        blob = cv.dnn.blobFromImage(
            image=frame,
            scalefactor=1.0 / 255.0,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        # blob을 네트워크 입력으로 설정
        self.net.setInput(blob)

        # 모든 YOLO 출력 헤드에 대해 순전파 실행
        try:
            outputs = self.net.forward(self.output_layer_names)
        except cv.error:
            # CUDA 경로가 실패하면 CPU 경로로 한 번 더 재시도
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            self.using_cuda = False
            self.backend_name = "cv.dnn.DNN_BACKEND_OPENCV"
            self.target_name = "cv.dnn.DNN_TARGET_CPU"
            outputs = self.net.forward(self.output_layer_names)

        # NMS 이전 원시 후보 검출을 저장할 컨테이너
        boxes_xywh: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        # 모든 출력 텐서를 순회하며 신뢰도 높은 대상 클래스만 유지
        for output in outputs:
            for detection in output:
                # YOLO 출력에서 클래스 확률 벡터 추출
                scores = detection[5:]

                # 가장 높은 신뢰도를 가진 클래스 선택
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # 신뢰도 낮거나 대상 클래스가 아니면 제외
                if confidence < self.conf_threshold or class_id not in TARGET_CLASS_IDS:
                    continue

                # 정규화된 중심 형식 박스를 픽셀 좌표로 변환
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                box_w = int(detection[2] * width)
                box_h = int(detection[3] * height)

                # NMSBoxes가 요구하는 좌상단 형식으로 변환
                x = int(center_x - box_w / 2)
                y = int(center_y - box_h / 2)

                # NMS 단계용 원시 후보를 저장
                boxes_xywh.append([x, y, box_w, box_h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # 후보가 없으면 빈 검출 배열 반환
        if len(boxes_xywh) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 겹침이 큰 중복 후보를 억제하기 위해 NMS 적용
        indices = cv.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.nms_threshold)

        # NMS 결과가 비면 빈 검출 배열 반환
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 최종 검출 텐서 [x1,y1,x2,y2,conf,class_id] 구성
        detections: List[List[float]] = []
        for idx in np.array(indices).reshape(-1):
            # NMS로 선택된 박스를 x,y,w,h 형식으로 읽기
            x, y, w, h = boxes_xywh[int(idx)]

            # SORT 입력 형식인 x1,y1,x2,y2로 변환
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + w)
            y2 = min(height - 1, y + h)

            # 신뢰도와 클래스 ID를 포함한 검출 행 저장
            detections.append([float(x1), float(y1), float(x2), float(y2), float(confidences[int(idx)]), float(class_ids[int(idx)])])

        # float32 numpy 배열로 반환
        return np.array(detections, dtype=np.float32)


def draw_tracks(frame: np.ndarray, tracks: List[KalmanTrack]) -> None:
    # 활성 트랙마다 사각형과 ID/클래스 라벨 렌더링
    for track in tracks:
        # 현재 바운딩 박스를 읽고 이미지 범위로 클리핑
        x1, y1, x2, y2 = track.get_state().astype(np.int32)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        # 바운딩 사각형 그리기
        cv.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

        # 안정적인 트랙 ID와 의미 있는 클래스명으로 라벨 구성
        class_name = COCO_NAME_MAP.get(track.class_id, str(track.class_id))
        label = f"ID {track.track_id} | {class_name} | {track.confidence:.2f}"

        # 가독성 높은 배경 박스를 위해 텍스트 크기 측정
        text_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 텍스트 대비 향상을 위해 채워진 배경 사각형 그리기
        cv.rectangle(
            frame,
            (x1, max(0, y1 - text_size[1] - 8)),
            (x1 + text_size[0] + 6, y1),
            (40, 220, 40),
            -1,
        )

        # 배경 사각형 위에 라벨 텍스트 그리기
        cv.putText(
            frame,
            label,
            (x1 + 3, y1 - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv.LINE_AA,
        )


def draw_detections(frame: np.ndarray, detections: np.ndarray) -> None:
    # 원시 후보 품질 점검을 위해 트래킹 전 검출 결과를 시각화
    for det in detections:
        # 정수형 바운딩 박스 모서리 좌표 읽기
        x1, y1, x2, y2 = det[:4].astype(np.int32)

        # 라벨 생성을 위한 클래스 ID와 신뢰도 읽기
        confidence = float(det[4])
        class_id = int(det[5])
        class_name = COCO_NAME_MAP.get(class_id, str(class_id))

        # 원시 검출 바운딩 박스 그리기
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 40), 2)

        # 검출 라벨 텍스트 구성
        label = f"DET {class_name} {confidence:.2f}"

        # 박스 위에 검출 라벨 텍스트 배치
        cv.putText(
            frame,
            label,
            (x1, max(12, y1 - 6)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 40),
            1,
            cv.LINE_AA,
        )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 런타임 시작 전 필수 입력 자산 존재 여부 확인
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not YOLO_CFG_PATH.exists():
        raise FileNotFoundError(f"YOLO cfg not found: {YOLO_CFG_PATH}")
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS_PATH}")

    # CUDA 불가 환경에서 빠르게 감지하기 위해 검출기를 먼저 초기화
    detector = YoloV3Detector(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)

    # 프레임 단위 처리를 위해 비디오 스트림 열기
    cap = cv.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    # 출력 비디오 라이터 설정을 위한 메타데이터 읽기
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv.CAP_PROP_FPS))
    fps = input_fps if input_fps > 0 else 30.0

    # 추적 시각화 결과를 저장할 출력 라이터 생성
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {OUTPUT_VIDEO_PATH}")

    # 교통 장면에 맞춘 기본값으로 SORT 트래커 구성
    tracker = SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)

    # 요약 파일에 기록할 실행 통계 변수 초기화
    frame_index = 0
    elapsed_total = 0.0
    seen_track_ids = set()
    last_frame = None

    # EOF 또는 사용자 중지 시점까지 프레임 반복 처리
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임마다 검출+추적을 수행하고 지연 시간 측정
        start_tick = cv.getTickCount()
        detections = detector.detect(frame)
        active_tracks = tracker.update(detections)
        end_tick = cv.getTickCount()

        # 프레임 처리 시간 통계 갱신
        elapsed = (end_tick - start_tick) / cv.getTickFrequency()
        elapsed_total += elapsed
        frame_index += 1

        # 이번 실행에서 생성된 고유 트랙 ID를 전역 집합에 누적
        for track in active_tracks:
            seen_track_ids.add(track.track_id)

        # 현재 프레임의 트랙 시각화 렌더링
        draw_tracks(frame, active_tracks)

        # 원시 검출 결과를 담은 대표 중간 프레임 저장
        if frame_index == 30:
            # 검출/트래킹 시각화가 섞이지 않도록 프레임 복사
            detection_frame = frame.copy()

            # 현재 검출값만 사용해 검출 전용 이미지 구성
            draw_detections(detection_frame, detections)

            # 검출 중간 결과 이미지를 파일로 저장
            cv.imwrite(str(OUTPUT_MID_DETECTION_PATH), detection_frame)

        # 트래킹 ID가 포함된 대표 중간 프레임 저장
        if frame_index == 60:
            # 트래킹 주석 프레임을 중간 결과로 저장
            cv.imwrite(str(OUTPUT_MID_TRACKING_PATH), frame)

        # 현재 프레임 기준 순간 처리 FPS 계산
        proc_fps = (1.0 / elapsed) if elapsed > 1e-6 else 0.0

        # 프레임 번호/활성 트랙/FPS 오버레이 렌더링
        cv.putText(
            frame,
            f"Frame: {frame_index}  Active Tracks: {len(active_tracks)}  Proc FPS: {proc_fps:.2f}",
            (10, 26),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 200, 255),
            2,
            cv.LINE_AA,
        )

        # 주석이 반영된 프레임을 출력 비디오에 기록
        writer.write(frame)

        # 최종 보고용으로 가장 최근 주석 프레임을 보관
        last_frame = frame.copy()

        # 실시간 추적 미리보기 창 표시
        cv.imshow("SORT Multi-Object Tracking (YOLOv3 CUDA)", frame)

        # 사용자가 q를 누르면 조기 종료 허용
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # 비디오 자원 해제 및 OpenCV 창 정리
    cap.release()
    writer.release()
    cv.destroyAllWindows()

    # 요약 파일용 집계 FPS 계산
    avg_proc_fps = (frame_index / elapsed_total) if elapsed_total > 1e-6 else 0.0

    # 최소 1프레임 처리 시 최종 스냅샷 저장
    if last_frame is not None:
        cv.imwrite(str(OUTPUT_FINAL_FRAME_PATH), last_frame)

    # 결과 경로와 통계를 담은 요약 텍스트 구성
    summary_lines = [
        "SORT Multi-Object Tracking Summary",
        f"Input video: {VIDEO_PATH}",
        f"YOLO cfg: {YOLO_CFG_PATH}",
        f"YOLO weights: {YOLO_WEIGHTS_PATH}",
        f"Output video: {OUTPUT_VIDEO_PATH}",
        f"Mid detection frame: {OUTPUT_MID_DETECTION_PATH}",
        f"Mid tracking frame: {OUTPUT_MID_TRACKING_PATH}",
        f"Final frame: {OUTPUT_FINAL_FRAME_PATH}",
        f"Processed frames: {frame_index}",
        f"Unique track ids: {len(seen_track_ids)}",
        f"Average processing FPS: {avg_proc_fps:.2f}",
        f"Detector backend: {detector.backend_name}",
        f"Detector target: {detector.target_name}",
    ]

    # 보고서 첨부용 요약 파일 저장
    OUTPUT_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    # 결과 파일 경로를 터미널에 출력
    print(f"Saved tracking video to: {OUTPUT_VIDEO_PATH}")
    print(f"Saved summary to: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    # 엔드투엔드 SORT 추적 파이프라인 실행
    main()
```

## 02. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

### 과제 설명
- 본 과제는 Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 실시간으로 추출하고 시각화하는 프로그램을 구현하는 것을 목표로 한다.
- OpenCV를 사용해 웹캠 영상을 프레임 단위로 캡처하고, 각 프레임을 FaceMesh 입력 형식인 RGB 이미지로 변환한 뒤 랜드마크를 검출한다.
- 검출된 랜드마크는 OpenCV의 `circle` 함수를 이용해 각 점으로 표시한다.
- ESC 키를 누르면 프로그램이 종료되도록 구성하여 실시간 데모 흐름을 완성한다.
- 입력 프레임 전처리는 CUDA 디바이스가 उपलब्ध한 경우 OpenCV CUDA 경로를 우선 시도해 GPU를 활용하도록 구성한다.

### 중간 결과물 (설명 포함)
- `week6/results_mediapipe/mediapipe_mid_landmarks_frame.jpg`
  - 웹캠 처리 중간 시점에서 저장되는 중간 결과 이미지이다.
  - 얼굴 랜드마크가 실시간으로 추출되고 점 형태로 시각화되는 상태를 확인할 수 있다.
- `week6/results_mediapipe/mediapipe_summary.txt`
  - 처리 프레임 수, 얼굴이 검출된 프레임 수, CUDA 전처리 사용 여부, 랜드마크 개수 정보가 기록된 요약 파일이다.

### 최종 결과물 (설명 포함)
- `week6/results_mediapipe/mediapipe_face_mesh_output.mp4`
  - 웹캠 전체 처리 과정이 저장된 최종 결과 영상이다.
  - 실시간 얼굴 랜드마크 시각화가 프레임 단위로 유지되는 모습을 확인할 수 있다.
- `week6/results_mediapipe/mediapipe_final_landmarks_frame.jpg`
  - 프로그램 종료 직전의 마지막 처리 프레임을 저장한 최종 스냅샷이다.
  - 마지막 시점의 랜드마크 분포와 화면 오버레이를 정적으로 확인할 수 있다.

### 코드 (주석 포함)
```python
from pathlib import Path
import importlib
import importlib.machinery
import importlib.util
import sys
import types

import cv2 as cv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VENV_ROOT = SCRIPT_DIR.parent
MEDIAPIPE_PACKAGE_DIR = VENV_ROOT / "Lib" / "site-packages" / "mediapipe"
RESULT_DIR = SCRIPT_DIR / "results_mediapipe"
OUTPUT_VIDEO_PATH = RESULT_DIR / "mediapipe_face_mesh_output.mp4"
OUTPUT_MID_FRAME_PATH = RESULT_DIR / "mediapipe_mid_landmarks_frame.jpg"
OUTPUT_FINAL_FRAME_PATH = RESULT_DIR / "mediapipe_final_landmarks_frame.jpg"
OUTPUT_SUMMARY_PATH = RESULT_DIR / "mediapipe_summary.txt"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30.0
MAX_FRAMES = 120
CUDA_AVAILABLE = hasattr(cv, "cuda") and cv.cuda.getCudaEnabledDeviceCount() > 0


def bootstrap_mediapipe_namespace() -> None:
    # 상위 mediapipe 패키지의 무거운 초기화를 피하기 위해 경로만 가진 가벼운 네임스페이스를 만든다.
    # 실제 mediapipe 소스 디렉터리를 패키지 검색 경로로 연결한다.
    def register_package(package_name: str, package_dir: Path) -> None:
        # 지정한 경로를 갖는 가벼운 패키지 모듈을 등록한다.
        package_module = types.ModuleType(package_name)
        package_module.__package__ = package_name
        package_module.__path__ = [str(package_dir)]

        # import machinery가 패키지로 인식할 수 있도록 스펙도 함께 만든다.
        package_spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package_spec.submodule_search_locations = [str(package_dir)]
        package_module.__spec__ = package_spec
        sys.modules[package_name] = package_module

    # 최상위 mediapipe 패키지를 등록한다.
    register_package("mediapipe", MEDIAPIPE_PACKAGE_DIR)

    # mediapipe.python 하위 패키지를 등록한다.
    register_package("mediapipe.python", MEDIAPIPE_PACKAGE_DIR / "python")

    # mediapipe.python.solutions 하위 패키지를 등록한다.
    register_package("mediapipe.python.solutions", MEDIAPIPE_PACKAGE_DIR / "python" / "solutions")

    # mediapipe.calculators 하위 패키지를 등록한다.
    register_package("mediapipe.calculators", MEDIAPIPE_PACKAGE_DIR / "calculators")


def load_face_mesh_module():
    # FaceMesh 소스 파일을 직접 로드해 패키지 import 문제를 우회한다.
    face_mesh_path = MEDIAPIPE_PACKAGE_DIR / "python" / "solutions" / "face_mesh.py"

    # 파일 위치로부터 모듈 스펙을 만든다.
    face_mesh_spec = importlib.util.spec_from_file_location(
        "mediapipe.python.solutions.face_mesh",
        face_mesh_path,
    )

    # 스펙이 유효하지 않으면 즉시 실패한다.
    if face_mesh_spec is None or face_mesh_spec.loader is None:
        raise ImportError(f"Failed to load FaceMesh module: {face_mesh_path}")

    # 스펙 기반 모듈 객체를 만든다.
    face_mesh_module = importlib.util.module_from_spec(face_mesh_spec)

    # 상대/절대 import가 참조할 수 있도록 sys.modules에 등록한다.
    sys.modules[face_mesh_spec.name] = face_mesh_module

    # 모듈 코드를 실제로 실행한다.
    face_mesh_spec.loader.exec_module(face_mesh_module)

    # 로드된 FaceMesh 모듈을 반환한다.
    return face_mesh_module


def preprocess_frame_for_facemesh(frame: cv.Mat) -> tuple[cv.Mat, bool]:
    # CUDA를 사용할 수 있으면 GPU에서 RGB 변환을 먼저 시도한다.
    if CUDA_AVAILABLE:
        try:
            # BGR 프레임을 GPU 메모리로 업로드한다.
            gpu_frame = cv.cuda_GpuMat()

            # 현재 프레임을 GPU 버퍼에 저장한다.
            gpu_frame.upload(frame)

            # MediaPipe가 기대하는 형식이 되도록 GPU에서 BGR을 RGB로 변환한다.
            gpu_rgb = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2RGB)

            # MediaPipe 처리를 위해 RGB 프레임을 호스트 메모리로 내려받는다.
            rgb_frame = gpu_rgb.download()

            # CUDA 경로가 성공적으로 사용되었음을 반환한다.
            return rgb_frame, True
        except cv.error:
            # CUDA 전처리가 불가능하면 CPU 변환으로 되돌아간다.
            pass

    # CUDA를 사용할 수 없으면 CPU에서 BGR을 RGB로 변환한다.
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB), False


def draw_face_mesh(frame: cv.Mat, face_landmarks, frame_width: int, frame_height: int) -> None:
    # 검출된 모든 얼굴을 순회하며 각 랜드마크 집합을 그린다.
    for landmarks in face_landmarks:
        # 정규화된 각 랜드마크를 작은 원으로 표시한다.
        for landmark in landmarks.landmark:
            # 정규화된 x 좌표를 픽셀 좌표로 변환한다.
            x = int(landmark.x * frame_width)

            # 정규화된 y 좌표를 픽셀 좌표로 변환한다.
            y = int(landmark.y * frame_height)

            # 화면 밖에 있는 랜드마크는 건너뛴다.
            if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
                continue

            # 현재 얼굴 랜드마크를 보이도록 점으로 그린다.
            cv.circle(frame, (x, y), 1, (0, 255, 0), -1, cv.LINE_AA)


def main() -> None:
    # mediapipe.solutions.face_mesh를 직접 불러오기 전에 네임스페이스를 준비한다.
    bootstrap_mediapipe_namespace()

    # FaceMesh 솔루션 모듈을 파일 경로로 직접 로드한다.
    mp_face_mesh = load_face_mesh_module()

    # 결과 저장 폴더가 없으면 생성한다.
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 실시간 캡처를 위해 웹캠 스트림을 연다.
    cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)

    # 카메라를 열 수 없으면 즉시 중단한다.
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {CAMERA_INDEX}")

    # 안정적인 랜드마크 시각화를 위해 일정한 프레임 크기를 요청한다.
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, TARGET_FPS)

    # 캡처 백엔드가 설정을 반영한 뒤 실제 해상도를 읽는다.
    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT
    actual_fps = float(cap.get(cv.CAP_PROP_FPS)) or TARGET_FPS

    # 최종 시각화를 저장할 비디오 라이터를 생성한다.
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, actual_fps, (actual_width, actual_height))

    # 출력 라이터를 만들 수 없으면 중단한다.
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output writer: {OUTPUT_VIDEO_PATH}")

    # 웹캠 실시간 스트림용 FaceMesh 검출기를 생성한다.
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 요약 생성을 위해 처리한 프레임 수를 추적한다.
    frame_index = 0

    # GPU 전처리 경로가 한 번이라도 사용되었는지 기록한다.
    used_cuda_preprocessing = False

    # 최종 스냅샷으로 사용할 마지막 주석 프레임을 보관한다.
    last_frame = None

    # 전체 세션에서 얼굴이 검출된 프레임 수를 센다.
    detected_face_frames = 0

    # 사용자가 ESC를 누르거나 최대 프레임 수에 도달할 때까지 웹캠 프레임을 처리한다.
    while True:
        # 카메라에서 프레임 하나를 읽는다.
        ret, frame = cap.read()

        # 카메라 스트림이 끝나면 루프를 종료한다.
        if not ret:
            break

        # 카메라의 BGR 프레임을 CUDA 또는 CPU 경로로 RGB로 변환한다.
        rgb_frame, used_cuda = preprocess_frame_for_facemesh(frame)

        # CUDA 전처리가 한 번이라도 사용되면 이를 기억한다.
        used_cuda_preprocessing = used_cuda_preprocessing or used_cuda

        # RGB 프레임에 대해 FaceMesh 추론을 수행한다.
        results = face_mesh.process(rgb_frame)

        # 검출된 얼굴 랜드마크를 BGR 프레임 위에 그린다.
        if results.multi_face_landmarks:
            # 얼굴이 하나 이상 검출되면 얼굴 존재 프레임 수를 증가시킨다.
            detected_face_frames += 1

            # 검출된 각 얼굴의 모든 랜드마크를 점으로 그린다.
            draw_face_mesh(frame, results.multi_face_landmarks, actual_width, actual_height)

            # 랜드마크가 검출되었음을 보여주는 짧은 오버레이를 표시한다.
            cv.putText(
                frame,
                f"FaceMesh landmarks: {len(results.multi_face_landmarks)} face(s)",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv.LINE_AA,
            )
        else:
            # 얼굴이 검출되지 않았을 때 대체 메시지를 표시한다.
            cv.putText(
                frame,
                "FaceMesh landmarks: none",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2,
                cv.LINE_AA,
            )

        # 프레임 주석 처리가 끝나면 처리 프레임 수를 증가시킨다.
        frame_index += 1

        # 자동 종료를 위해 최대 프레임 수에 도달하면 루프를 끝낸다.
        if frame_index >= MAX_FRAMES:
            break

        # 보고서용 중간 스냅샷을 저장한다.
        if frame_index == 30:
            # 현재 주석 프레임을 중간 결과 이미지로 저장한다.
            cv.imwrite(str(OUTPUT_MID_FRAME_PATH), frame)

        # 현재 주석 프레임을 최종 스냅샷 후보로 보관한다.
        last_frame = frame.copy()

        # 주석이 포함된 프레임을 출력 비디오에 기록한다.
        writer.write(frame)

        # 실시간 랜드마크 시각화를 화면에 표시한다.
        cv.imshow("Mediapipe FaceMesh (CUDA-preprocessed when available)", frame)

        # ESC 키가 눌리면 즉시 종료한다.
        if cv.waitKey(1) & 0xFF == 27:
            break

    # 루프가 끝나면 카메라와 라이터 자원을 해제한다.
    cap.release()
    writer.release()

    # 열려 있는 OpenCV 창을 모두 닫는다.
    cv.destroyAllWindows()

    # 프레임이 하나라도 처리되었다면 최종 주석 프레임을 저장한다.
    if last_frame is not None:
        cv.imwrite(str(OUTPUT_FINAL_FRAME_PATH), last_frame)

    # 보고서용 간단한 요약 텍스트를 구성한다.
    summary_lines = [
        "Mediapipe FaceMesh Summary",
        f"Camera index: {CAMERA_INDEX}",
        f"Output video: {OUTPUT_VIDEO_PATH}",
        f"Mid frame: {OUTPUT_MID_FRAME_PATH}",
        f"Final frame: {OUTPUT_FINAL_FRAME_PATH}",
        f"Processed frames: {frame_index}",
        f"Max frames: {MAX_FRAMES}",
        f"Face-present frames: {detected_face_frames}",
        f"CUDA preprocessing: {'enabled' if used_cuda_preprocessing else 'disabled'}",
        f"OpenCV CUDA devices: {cv.cuda.getCudaEnabledDeviceCount() if hasattr(cv, 'cuda') else 0}",
        "FaceMesh module: mediapipe.python.solutions.face_mesh.FaceMesh",
        "Landmark count per face: 468",
    ]

    # 과제 보고서용 요약 파일을 저장한다.
    OUTPUT_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    # 생성된 결과 파일을 터미널에 출력한다.
    print(f"Saved tracking video to: {OUTPUT_VIDEO_PATH}")
    print(f"Saved summary to: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    # 실시간 FaceMesh 파이프라인을 실행한다.
    main()
```




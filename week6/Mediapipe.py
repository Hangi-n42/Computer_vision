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
    # CUDA를 사용할 수 있으면 GPU에서 먼저 RGB 변환을 시도한다.
    if CUDA_AVAILABLE:
        try:
            # BGR 프레임을 GPU 메모리로 업로드한다.
            gpu_frame = cv.cuda_GpuMat()

            # 현재 프레임을 GPU 버퍼에 저장한다.
            gpu_frame.upload(frame)

            # MediaPipe가 기대하는 형식으로 맞추기 위해 GPU에서 BGR을 RGB로 변환한다.
            gpu_rgb = cv.cuda.cvtColor(gpu_frame, cv.COLOR_BGR2RGB)

            # MediaPipe 처리를 위해 RGB 프레임을 호스트 메모리로 다시 내려받는다.
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

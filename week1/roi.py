import cv2 as cv
import numpy as np

drawing = False  # 드래그 중인지 여부
start_point = None  # 시작점
end_point = None  # 끝점
roi_selected = False  # ROI 선택 완료 여부
original_image = None  # 원본 이미지
display_image = None  # 화면에 표시할 이미지

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수"""
    global drawing, start_point, end_point, roi_selected, display_image
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 좌클릭 - 시작점 설정
        drawing = True
        roi_selected = False
        start_point = (x, y)
        end_point = (x, y)
        
    elif event == cv.EVENT_MOUSEMOVE:
        
        # 드래그 중 - 사각형 그리기
        if drawing:
            end_point = (x, y)
            # 원본 이미지 복사하여 사각형 그리기
            display_image = original_image.copy()
            cv.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            
    elif event == cv.EVENT_LBUTTONUP:
        # 마우스 버튼을 떼면 ROI 선택 완료
        drawing = False
        roi_selected = True
        end_point = (x, y)
        # 최종 사각형 그리기
        display_image = original_image.copy()
        cv.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
        
        # ROI 영역 추출 및 표시
        extract_and_show_roi()

def extract_and_show_roi():
    """ROI 영역을 추출하고 별도 창에 표시"""
    global start_point, end_point, original_image
    
    if start_point is None or end_point is None:
        return
    
    # 좌표 정렬 (왼쪽 위, 오른쪽 아래)
    x1 = min(start_point[0], end_point[0])
    y1 = min(start_point[1], end_point[1])
    x2 = max(start_point[0], end_point[0])
    y2 = max(start_point[1], end_point[1])
    
    # 유효한 영역인지 확인
    if x2 > x1 and y2 > y1:
        # numpy 슬라이싱으로 ROI 추출
        roi = original_image[y1:y2, x1:x2]
        
        if roi.size > 0:
            cv.imshow('ROI', roi)
            print(f"ROI 선택 완료: ({x1}, {y1}) ~ ({x2}, {y2})")

def reset_selection():
    """영역 선택 리셋"""
    global start_point, end_point, roi_selected, display_image
    start_point = None
    end_point = None
    roi_selected = False
    display_image = original_image.copy()
    cv.destroyWindow('ROI')
    print("영역 선택이 리셋되었습니다.")

def save_roi():
    """선택한 ROI를 파일로 저장"""
    global start_point, end_point, original_image, roi_selected
    
    if not roi_selected or start_point is None or end_point is None:
        print("저장할 ROI가 선택되지 않았습니다.")
        return
    
    # 좌표 정렬
    x1 = min(start_point[0], end_point[0])
    y1 = min(start_point[1], end_point[1])
    x2 = max(start_point[0], end_point[0])
    y2 = max(start_point[1], end_point[1])
    
    if x2 > x1 and y2 > y1:
        # ROI 추출
        roi = original_image[y1:y2, x1:x2]
        
        if roi.size > 0:
            # 파일 저장
            filename = 'roi_selected.jpg'
            cv.imwrite(filename, roi)
            print(f"ROI가 '{filename}'로 저장되었습니다.")
        else:
            print("저장할 영역이 비어있습니다.")
    else:
        print("유효하지 않은 영역입니다.")

# 이미지 로드
original_image = cv.imread('soccer.jpg')
if original_image is None:
    print("오류: soccer.jpg 파일을 찾을 수 없습니다.")
    exit()

display_image = original_image.copy()

# 윈도우 생성 및 마우스 콜백 설정
window_name = 'Image ROI Selection'
cv.namedWindow(window_name)
cv.setMouseCallback(window_name, mouse_callback)

print("=== ROI 선택 프로그램 ===")
print("마우스 클릭 & 드래그: ROI 영역 선택")
print("r: 영역 선택 리셋")
print("s: 선택한 영역을 이미지 파일로 저장")
print("q: 종료")

while True:
    cv.imshow(window_name, display_image)
    
    # 키 입력 대기 (1ms)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # q 키를 누르면 종료
        print("프로그램을 종료합니다.")
        break
    elif key == ord('r'):
        # r 키를 누르면 영역 선택 리셋
        reset_selection()
    elif key == ord('s'):
        # s 키를 누르면 선택한 영역 저장
        save_roi()

cv.destroyAllWindows()
